from traceback import print_tb
import torch, pdb
import torch.nn as nn

from transformers import AdamW, get_linear_schedule_with_warmup
from models.util.slconfig import SLConfig
from utils import util

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import socket
from collections import OrderedDict
from pathlib import Path
from typing import *
from utils.loadData import *
from utils.eval import *
from eval_tools import AveragePrecisionMeter
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from peft import LoraConfig, get_peft_model, PeftModel
import numpy as np
from math import ceil
from typing import List, Optional
from dynamic_weights import *
from torch.utils.data import DataLoader

from math import ceil
from typing import List, Optional, Dict
import numpy as np
from torch.utils.data import DataLoader

class MultiTaskLoader1:
    """
    一个可迭代的“多任务” DataLoader 适配器。
    - 支持两种模式：
        1) mode='power' : 温度/幂次采样，p_i ∝ sizes[i] ** alpha
           * 若提供 min_ratio/min_steps，则先满足下限，再按幂采样补足，并在 epoch 级构建日程。
           * 若未提供下限，则保持逐步随机抽样（与你原实现一致）。
        2) mode='ratio' : 定比混采，按给定比例拼接周期表
    - 产出 (task_id, batch)，可与 enumerate(train_loader) 无缝配合
    """

    def __init__(
        self,
        task_loaders: List[DataLoader],
        mode: str = "power",
        steps_per_epoch: Optional[int] = None,
        *,
        # power 模式参数
        sizes: Optional[List[int]] = None,
        alpha: float = 0.4,
        # ↓↓↓ 新增：幂采样的下限控制（二选一或都可给）
        min_ratio: Optional[Dict[int, float]] = None,  # 例如 {1: 0.30} 表示任务1至少30%
        min_steps: Optional[Dict[int, int]] = None,    # 例如 {1: 900}  表示任务1至少900步
        # ratio 模式参数
        ratio: Optional[List[int]] = None,
        reshuffle_each_epoch: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            task_loaders: 每个任务各自的 PyTorch DataLoader（建议各自 shuffle/drop_last 设置好）
            mode: 'power' 或 'ratio'
            steps_per_epoch: 每个 epoch 迭代多少步；None 表示无限（不建议与 enumerate 搭配）
            sizes: (power) 每个任务样本量，用于计算采样概率
            alpha: (power) 幂次/温度系数，0~1 常用
            min_ratio/min_steps: (power) 对特定任务施加最少占比/步数的下限
            ratio: (ratio) 比例表，例如 [1,2] 表示 task0:task1=1:2
            reshuffle_each_epoch: (ratio) 每个 epoch 是否打乱拼接后的日程表
            seed: 随机种子
        """
        assert len(task_loaders) >= 2, "需要至少两个任务"
        self.task_loaders = task_loaders
        self.num_tasks = len(task_loaders)
        self.mode = mode
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # 为每个子 DataLoader 准备一个迭代器
        self._iters = [iter(dl) for dl in self.task_loaders]

        # —— power 初始化
        self.min_ratio = (min_ratio or {})
        self.min_steps = (min_steps or {})
        self._power_has_min = (mode == "power") and (len(self.min_ratio) > 0 or len(self.min_steps) > 0)

        if mode == "power":
            assert sizes is not None and len(sizes) == self.num_tasks, "power 模式需提供 sizes"
            sizes = np.asarray(sizes, dtype=np.float64)
            weights = np.power(sizes, float(alpha))
            self._p = (weights / weights.sum()).astype(np.float64)

        # —— ratio 初始化
        elif mode == "ratio":
            assert ratio is not None and len(ratio) == self.num_tasks, "ratio 模式需提供 ratio"
            assert any(k > 0 for k in ratio), "ratio 需至少一个正数"
            self._base_schedule = []
            for tid, k in enumerate(ratio):
                self._base_schedule += [tid] * int(k)
            self._reshuffle = bool(reshuffle_each_epoch)
        else:
            raise ValueError("mode 必须是 'power' 或 'ratio'")

        # 迭代状态
        self._built = False
        self._epoch_steps_left: Optional[int] = None
        self._epoch_schedule: Optional[List[int]] = None
        self._ptr = 0

    def _reset_sub_iter(self, task_id: int):
        """当某个子 loader 耗尽时，重建其迭代器（DataLoader 本身是可重入的）。"""
        self._iters[task_id] = iter(self.task_loaders[task_id])

    # ===== 新增：幂采样 + 下限约束 的日程构建 =====
    def _build_power_schedule_with_min(self, T: int) -> List[int]:
        """
        返回长度为 T 的任务ID列表：
          1) 先满足 min_steps / min_ratio 下限
          2) 再按幂采样概率 self._p 抽取剩余步数
          3) 打乱
        """
        min_counts = np.zeros(self.num_tasks, dtype=int)
        # min_ratio
        for tid, r in self.min_ratio.items():
            min_counts[tid] = max(min_counts[tid], int(np.floor(T * float(r))))
        # min_steps
        for tid, k in self.min_steps.items():
            min_counts[tid] = max(min_counts[tid], int(k))

        total_min = int(min_counts.sum())
        if total_min > T:
            raise ValueError(f"下限配额之和 {total_min} > steps_per_epoch {T}，请下调 min_ratio/min_steps。")

        remain = T - total_min
        counts = min_counts.copy()
        if remain > 0:
            extra = self.rng.choice(self.num_tasks, size=remain, p=self._p).astype(int)
            for tid in extra:
                counts[tid] += 1

        schedule = []
        for tid, k in enumerate(counts.tolist()):
            schedule += [tid] * int(k)
        self.rng.shuffle(schedule)
        return schedule

    def _build_epoch(self):
        """根据 mode 构建本 epoch 的调度计划，并设置步数。"""
        if self.steps_per_epoch is None:
            # 提醒：无限模式不适合 enumerate；这里设置一个大数仅防止误用。
            self._epoch_steps_left = 10**18
        else:
            self._epoch_steps_left = int(self.steps_per_epoch)

        if self.mode == "ratio":
            base = self._base_schedule
            rep = ceil(self._epoch_steps_left / len(base))
            full = (base * rep)[: self._epoch_steps_left]
            if self._reshuffle:
                self.rng.shuffle(full)
            self._epoch_schedule = full
            self._ptr = 0

        elif self.mode == "power" and self._power_has_min:
            # 有下限约束时，必须已知 steps_per_epoch，用整轮日程来保证下限
            if self._epoch_steps_left >= 10**18:
                raise ValueError("power+min_ratio/min_steps 模式下必须设置 steps_per_epoch。")
            self._epoch_schedule = self._build_power_schedule_with_min(self._epoch_steps_left)
            self._ptr = 0

        # 若为 power 且无下限，则不预建 schedule，按步抽签（保留你原逻辑）
        self._built = True

    def __iter__(self):
        # 每次进入新一轮迭代都视作新 epoch
        self._built = False
        self._build_epoch()
        return self

    def __next__(self):
        if not self._built:
            self._build_epoch()

        if self._epoch_steps_left <= 0:
            raise StopIteration

        # 选任务
        if self.mode == "power":
            if self._power_has_min:
                # 用整轮日程
                task_id = self._epoch_schedule[self._ptr]
                self._ptr += 1
            else:
                # 逐步随机抽签（原版）
                task_id = int(self.rng.choice(self.num_tasks, p=self._p))
        else:  # ratio
            task_id = self._epoch_schedule[self._ptr]
            self._ptr += 1

        # 拿 batch（子 loader 用尽就重置）
        try:
            batch = next(self._iters[task_id])
        except StopIteration:
            self._reset_sub_iter(task_id)
            batch = next(self._iters[task_id])

        self._epoch_steps_left -= 1
        return task_id, batch



class MultiTaskLoader:
    """
    一个可迭代的“多任务” DataLoader 适配器。
    - 支持两种模式：
        1) mode='power' : 温度/幂次采样，p_i ∝ sizes[i] ** alpha
        2) mode='ratio' : 定比混采，按给定比例拼接周期表
    - 产出 (task_id, batch)，可与 enumerate(train_loader) 无缝配合
    """

    def __init__(
        self,
        task_loaders: List[DataLoader],
        mode: str = "power",
        steps_per_epoch: Optional[int] = None,
        *,
        # power 模式参数
        sizes: Optional[List[int]] = None,
        alpha: float = 0.4,
        # ratio 模式参数
        ratio: Optional[List[int]] = None,
        reshuffle_each_epoch: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            task_loaders: 每个任务各自的 PyTorch DataLoader（建议各自 shuffle/drop_last 设置好）
            mode: 'power' 或 'ratio'
            steps_per_epoch: 每个 epoch 迭代多少步；None 表示无限（不建议与 enumerate 搭配）
            sizes: (power) 每个任务样本量，用于计算采样概率
            alpha: (power) 幂次/温度系数，0~1 常用
            ratio: (ratio) 比例表，例如 [1,2] 表示 task0:task1=1:2
            reshuffle_each_epoch: (ratio) 每个 epoch 是否打乱拼接后的日程表
            seed: 随机种子
        """
        assert len(task_loaders) >= 2, "需要至少两个任务"
        self.task_loaders = task_loaders
        self.num_tasks = len(task_loaders)
        self.mode = mode
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # 为每个子 DataLoader 准备一个迭代器
        self._iters = [iter(dl) for dl in self.task_loaders]

        if mode == "power":
            assert sizes is not None and len(sizes) == self.num_tasks, "power 模式需提供 sizes"
            sizes = np.asarray(sizes, dtype=np.float64)
            weights = np.power(sizes, float(alpha))
            self._p = (weights / weights.sum()).astype(np.float64)
        elif mode == "ratio":
            assert ratio is not None and len(ratio) == self.num_tasks, "ratio 模式需提供 ratio"
            assert any(k > 0 for k in ratio), "ratio 需至少一个正数"
            self._base_schedule = []
            for tid, k in enumerate(ratio):
                self._base_schedule += [tid] * int(k)
            self._reshuffle = bool(reshuffle_each_epoch)
        else:
            raise ValueError("mode 必须是 'power' 或 'ratio'")

        # 迭代状态
        self._built = False
        self._epoch_steps_left = None
        self._epoch_schedule = None
        self._ptr = 0

    def _reset_sub_iter(self, task_id: int):
        """当某个子 loader 耗尽时，重建其迭代器（DataLoader 本身是可重入的）。"""
        self._iters[task_id] = iter(self.task_loaders[task_id])

    def _build_epoch(self):
        """根据 mode 构建本 epoch 的调度计划（仅 ratio 需要），并设置步数。"""
        if self.steps_per_epoch is None:
            # 提醒：无限模式不适合 enumerate；这里设置一个大数仅防止误用。
            self._epoch_steps_left = 10**18
        else:
            self._epoch_steps_left = int(self.steps_per_epoch)

        if self.mode == "ratio":
            base = self._base_schedule
            rep = ceil(self._epoch_steps_left / len(base))
            full = (base * rep)[: self._epoch_steps_left]
            if self._reshuffle:
                self.rng.shuffle(full)
            self._epoch_schedule = full
            self._ptr = 0

        self._built = True

    def __iter__(self):
        # 每次进入新一轮迭代都视作新 epoch
        self._built = False
        self._build_epoch()
        return self

    def __next__(self):
        if not self._built:
            self._build_epoch()

        if self._epoch_steps_left <= 0:
            raise StopIteration

        # 选任务
        if self.mode == "power":
            task_id = int(self.rng.choice(self.num_tasks, p=self._p))
        else:  # ratio
            task_id = self._epoch_schedule[self._ptr]
            self._ptr += 1

        # 拿 batch（子 loader 用尽就重置）
        try:
            batch = next(self._iters[task_id])
        except StopIteration:
            self._reset_sub_iter(task_id)
            batch = next(self._iters[task_id])

        self._epoch_steps_left -= 1
        return task_id, batch




def trainer(model, args):

    log = util.get_logger(args.record_dir, "root", "debug")
    util.set_seed(args.seed)
    torch.cuda.set_device(int(args.device[-1]))
    torch.backends.cudnn.benchmark = True
    tokenizer = model.tokenizer
    device = torch.cuda.current_device()
    model = model.to(device)

    train_loader_sc, _ , _ = get_dataloaders(args, tokenizer = tokenizer, batch_size=args.batch_size, num_train=args.num_train, num_val=args.num_val, num_workers=args.num_workers, SC=True)
    train_loader_df, dev_loader_df, test_loader_df = get_dataloaders(args, tokenizer = tokenizer, batch_size=args.batch_size, num_train=args.num_train, num_val=args.num_val, num_workers=args.num_workers)

    num_train = len(train_loader_df.dataset)+len(train_loader_sc.dataset) # 208184+3546
    # num_val = len(dev_loader.dataset)     # 22126
    # num_test = len(test_loader.dataset)   # 50705
    each_step = (num_train // args.batch_size) # 34698+591
    total_steps = ( (num_train // args.batch_size) * args.epochs)     # num times that optim.step() will be called
    total_train = num_train * args.epochs
    print('data loaded, start training')

    # 小:大 = 1:2
    steps_per_epoch = 50000
    # steps_per_epoch = 1000
    dl_small = train_loader_sc
    dl_big = train_loader_df
    train_loader = MultiTaskLoader(task_loaders=[dl_small, dl_big], mode="ratio",
                     steps_per_epoch=steps_per_epoch, ratio=[2, 1], reshuffle_each_epoch=True,)
                #    steps_per_epoch=steps_per_epoch, ratio=[1, 3], reshuffle_each_epoch=True,)


    # train_loader = MultiTaskLoader1(task_loaders=[dl_small, dl_big], mode="ratio", steps_per_epoch=steps_per_epoch
    #                 , ratio=[3, 1], reshuffle_each_epoch=True, min_ratio={1: 0.30})

    # train_loader = MultiTaskLoader(task_loaders=[dl_big, dl_small], mode="power",
    #                 steps_per_epoch=steps_per_epoch, sizes=[208184, 3546], alpha=0.4, seed=2025,)
    # train_loader = MultiTaskLoader(task_loaders=[dl_big, dl_small], mode="power", steps_per_epoch=steps_per_epoch,
    #                 sizes=[208184, 3546], alpha=0.35, min_ratio={1: 0.30})

    num_train = steps_per_epoch


    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=args.adam_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    import shutil
    shutil.copyfile('/data1/zc/24forensics/CofiPara-master/trainer_u.py', os.path.join(args.save_path, 'trainer_u.py'))
    shutil.copyfile('/data1/zc/24forensics/CofiPara-master/model_config.py', os.path.join(args.save_path, 'model_config.py'))
 

   
   
    log.info(f'device: {device}\n'
            f'total_steps: {total_steps}\n'
            f'total_train (num_t * epoch): {total_train}\n'
            f'learning rate: {args.lr}\n'
            # f'save_dir: {save_path}\n'
            f'machine: {socket.gethostname()}\n')

    skip_training = args.skip_training
    if skip_training:
        print("Skip training...")
        args.epochs = 1
    # 'uncertainty' | 'gradnorm' | 'dwa'
    # gradnorm need shared_params
    # weighter = make_weighter("uncertainty", num_tasks=2).to(device)
    # weighter = make_weighter("gradnorm", num_tasks=2, shared_params=model.parameters(), alpha=0.5, lr=args.lr)
    weighter = make_weighter("dwa", num_tasks=2, T=2.0)


    epoch =  0      # number of times we have passed through entire set of training examples
    step = 0        # number of total examples we have done (will be epoch * len(data_set) at end of each epoch)
    while epoch < args.epochs:
        torch.cuda.empty_cache()
        epoch += 1
        model.train()
        running = torch.zeros(2)
        count = 0

        with torch.enable_grad(), tqdm(total=num_train*args.batch_size) as progress_bar:
            for batch_num, (task_id, batch) in enumerate(train_loader):
            # for batch_num, batch in enumerate(train_loader_sc):
                batch_size = len(batch["image_ids"])            
                # loss, _ = forward_u(model, device, batch, task_id=task_id)

                L_task, _ = forward_u(model, device, batch, task_id=task_id)
                losses = [torch.tensor(0., device=L_task.device),
                            torch.tensor(0., device=L_task.device)]
                losses[task_id] = L_task
                # loss, _ = weighter.combine_step(losses)
                loss, _ = weighter.combine_step_equal(losses)
              
                
                optimizer.zero_grad()

                loss.backward()
                running[task_id] += float(L_task.detach())
                count += 1
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()        # don't need to pass step to scheduler

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                        loss= loss.item())
        log.info(f'Saving model at epoch {epoch}...\n')
        
        
        model.save_state_dict(args, epoch)
        if isinstance(weighter, DWAWeighter):
            avg_losses = (running / max(1, count)).tolist()
            weighter.update_epoch(avg_losses)



def test(model,args):
    log = util.get_logger(args.record_dir, "root", "debug")
    util.set_seed(args.seed)
    torch.cuda.set_device(int(args.device[-1]))

    tokenizer = model.tokenizer
    model.cuda()
    device = torch.cuda.current_device()

    # train_loader, dev_loader, test_loader = \
    #     get_dataloaders(args, tokenizer = tokenizer, batch_size=args.batch_size, num_train=args.num_train, num_val=args.num_val,
    #                     num_workers=args.num_workers)
    train_loader, dev_loader, test_loader = get_dataloaders(args, tokenizer = tokenizer, batch_size=args.batch_size, num_train=args.num_train, num_val=args.num_val, num_workers=args.num_workers, SC=True)

    # test_loader = train_loader    # for dev eval

    # reset in case we used the -1 flag for all
    num_train = len(train_loader.dataset)
    num_test = len(test_loader.dataset)
    total_steps = ( (num_train // args.batch_size) * args.epochs)     # num times that optim.step() will be called
    total_train = num_train * args.epochs
    print('data loaded, start testing')



    log.info(f'device: {device}\n'
            f'total_steps: {total_steps}\n'
            f'total_train (num_t * epoch): {total_train}\n'
            f'learning rate: {args.lr}\n'
            f'model_dir: {args.checkpoint}\n'
            f'machine: {socket.gethostname()}\n')

    skip_training = args.skip_training
    if skip_training:
        print("Skip training...")
        args.epochs = 1      

    torch.cuda.empty_cache()
    model.eval()        # put model in eval mode
    

 

    ###############
    # Test (and to save results)
    ###############
    pred_list_all = []
    pred_list = []
    target_list = []
    EM = []
    box_res = []
    id_list = []
    target_boxes = []
    targets = []
    cls2_labels = []
    cls2_preds = []
    logits_multis = []
    y_pred = []
    y_true= []
    cls_nums_all, cls_acc_all= 0, 0
    em, em_sum = 0, 0

    with torch.no_grad(), tqdm(total=num_test) as progress_bar:
        multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
        multi_label_meter.reset()

        for batch_num, batch in enumerate(test_loader):           
            batch_size = len(batch["image_ids"])
            logits,tgt_ids,img_out, _, _ = forward_test_cls(model,tokenizer,device,batch)           
                      
            cls_label = tgt_ids#.view(-1)
            cls_label[cls_label == 1] = 0
            cls_label[cls_label == -100] = 0
            cls_label[cls_label == 332] = 1  # 'T', Real text
            cls_label[cls_label == 411] = 2  # 'O', Fake text

            for idx in range(cls_label.shape[0]):
                index_txt = (cls_label[idx] != 0)
                gts = cls_label[idx][index_txt]-1
                pred = torch.zeros(gts.shape)
                pred[torch.argmax(logits[idx].view(-1, 3)[index_txt], dim=1) == 2]=1
                if (gts.cpu()==pred).sum() == len(gts):
                    EM.append(1)
                else:
                    EM.append(0)
            pred_list.extend(pred.cpu())
            target_list.extend(gts.cpu())

            face_id_list = []
            for out, box in zip(img_out['pred_boxes'],batch['bboxes']):
                mask = box > 0
                length = int(len(box[mask])/4)
                if length != 0:
                    out = out[:1].cpu().detach().numpy()
                    box = box[:1].cpu().detach().numpy()
                    # box_res.append([out.tolist()[0]])
                    # target_boxes.append(box.tolist()[0])  
                    box_res.append(out)
                    target_boxes.append(box)      
            progress_bar.update(batch_size)
         

    
        _, f1, p, r, _ = evaluate_text(pred_list,target_list)
        EM = np.mean(EM)
        util.pred_to_csv(face_id_list,box_res,target_boxes)

        # iou, iou50, iou75 = eval_iou_batch(torch.tensor(box_res).squeeze(dim=1), torch.tensor(target_boxes).squeeze(dim=1))
        iou, iou50, iou75 =  eval_ap_batch(torch.tensor(box_res), torch.tensor(target_boxes))
        log.info(f'Test AP: {iou}, AP 50: {iou50}, AP 75: {iou75}\n')
        log.info(f'Test F1: {f1}, EM: {EM}, Recall: {r}, Precision: {p}')
       


def test_single(model,args):
    log = util.get_logger(args.record_dir, "root", "debug")
    util.set_seed(args.seed)
    torch.cuda.set_device(int(args.device[-1]))

    tokenizer = model.tokenizer
    model.cuda()
    device = torch.cuda.current_device()

    train_loader, dev_loader, test_loader = \
        get_dataloaders(args, tokenizer = tokenizer, batch_size=args.batch_size, num_train=args.num_train, num_val=args.num_val,
                        num_workers=args.num_workers)
    # test_loader = train_loader    # for dev eval

    # reset in case we used the -1 flag for all
    num_train = len(train_loader.dataset)
    num_test = len(test_loader.dataset)
    total_steps = ( (num_train // args.batch_size) * args.epochs)     # num times that optim.step() will be called
    total_train = num_train * args.epochs
    print('data loaded, start testing')



    log.info(f'device: {device}\n'
            f'total_steps: {total_steps}\n'
            f'total_train (num_t * epoch): {total_train}\n'
            f'learning rate: {args.lr}\n'
            f'model_dir: {args.checkpoint}\n'
            f'machine: {socket.gethostname()}\n')

    skip_training = args.skip_training
    if skip_training:
        print("Skip training...")
        args.epochs = 1

    epoch = 0       # number of times we have passed through entire set of training examples
    step = 0        # number of total examples we have done (will be epoch * len(data_set) at end of each epoch)
    best_acc = 0
    best_f1 = 0

    torch.cuda.empty_cache()
    
    # from peft import PeftModel, PeftConfig
    # model = PeftModel.from_pretrained(model, args.lora_path)
    # model = model.merge_and_unload()

    log.info(f'Evaluating at step {step}...')
    model.eval()        # put model in eval mode
    

    # See how the model is doing with exact match on tokens
    pred_list_all = []                      # accumulate for saving; list; one list per epoch

    ###############
    # Test (and to save results)
    ###############
    pred_list_all = []
    pred_list = []
    target_list = []
    EM = []
    box_res = []
    id_list = []
    target_boxes = []
    targets = []
    cls2_labels = []
    cls2_preds = []
    logits_multis = []
    y_pred = []
    y_true= []
    cls_nums_all, cls_acc_all= 0, 0

    with torch.no_grad(), tqdm(total=num_test) as progress_bar:
        multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
        multi_label_meter.reset()

        for batch_num, batch in enumerate(test_loader):
            # pdb.set_trace()
            batch_size = len(batch["image_ids"])
            logits,tgt_ids,img_out, logits_multi, logits_bi = forward_test_cls(model,device,batch)
            # pdb.set_trace()
            orig_text_output = batch["target_text"]
            if args.model_type != 't5': # false
                tgt_ids = tgt_ids.argmax(dim=-1)
                orig_text_output = [int(i) for i in tgt_ids]
          
            # if False not in (gts.cpu() == pred) and  batch['fake_cls'][0] =='face_attribute&text_attribute':
            # if '&' in batch['fake_cls'][0]:
            #     print(gts.cpu() == pred)
            #     print(gts)
            #     print(pred)
            #     pdb.set_trace()
            # cls for binary and multicls
           
            # ----- multi metrics -----
            # target, _= model.get_multi_label(batch['fake_cls'])
            target, _, _, _ = model.get_multi_label(batch['fake_cls'])

            # pdb.set_trace()
            # ##================= real/fake cls ========================##
            # cls_label = torch.ones(len(batch['fake_cls']), dtype=torch.long).to(device)
            # real_label_pos = np.where(np.array(batch['fake_cls']) == 'orig')[0].tolist()
            # cls_label[real_label_pos] = 0

            # cls_label = torch.ones(target.shape[0], dtype=torch.long).to(device)
            # cls_label = cls_label - target[:,-1].to(device) - target[:,-2].to(device) # for st
            cls_label = target[:,-1].to(device) + target[:,-2].to(device)  # for st
            # cls_label = target[:,0].to(device) + target[:,1].to(device) # for df
            # pdb.set_trace()

            y_pred.extend(F.softmax(logits_bi,dim=1)[:,1].cpu().flatten().tolist())
            # y_pred.extend(F.sigmoid(logits_bi[:,0]).cpu().flatten().tolist())
            y_true.extend(cls_label.cpu().flatten().tolist())

            pred_acc = logits_bi.argmax(1)
            cls_nums_all += cls_label.shape[0]
            cls_acc_all += torch.sum(pred_acc == cls_label).item()



            multi_label_meter.add(logits_multi, target)
            sigmoid = nn.Sigmoid()
            # print(sigmoid(logits_multi))
            # # print(logits_multi.cpu().tolist())
            # pdb.set_trace()
            logits_multis.extend(logits_multi.cpu().tolist())
            targets.extend(target.cpu().tolist())
            # print(logits_multis)
            # pdb.set_trace()


            if args.stage == "stage2":
                # pdb.set_trace()
                cls_label = tgt_ids.view(-1)
                cls_label[cls_label == 1] = 0
                cls_label[cls_label == -100] = 0
                cls_label[cls_label == 332] = 1  # 'T', Real text
                cls_label[cls_label == 411] = 2  # 'O', Fake text
                gts = cls_label[cls_label != 0]-1
                pred = torch.zeros(gts.shape)
                pred[torch.argmax(logits.view(-1, 3)[cls_label != 0], dim=1) == 2]=1
                # pdb.set_trace()
                batch_EM = 0

                # pred, gts , batch_acc, batch_EM = batch_post_process_dgm4(batch['source_text'], batch['target_text'], outputs_decoded)

                EM.append(batch_EM)
                ids = [ids[28:-4] for ids in batch["image_ids"]]
                id_list.extend(ids)
                pred_list.extend(pred.cpu())
                target_list.extend(gts.cpu())

                face_id_list = []
                for out, box, face_id in zip(img_out['pred_boxes'],batch['bboxes'], id_list):
                    mask = box > 0
                    length = int(len(box[mask])/4)
                    if length != 0:
                        out = out[:length]#.cpu().detach().numpy()
                        box = box[:length]#.cpu().detach().numpy()
                        # pdb.set_trace()
                        box_res.append([out.tolist()[0]])
                        face_id_list.append(face_id)
                        target_boxes.append(box.tolist()[0])


                # for batchID in range(len(batch['image_ids'])):
                #     if batch['fake_cls'][batchID] == 'orig':
                #         cls2_labels.append(0)
                #     else:
                #         cls2_labels.append(1)

                #     if img_out['pred_boxes'][batchID].sum() > 0:
                #         cls2_preds.append(1)
                #     else:
                #         cls2_preds.append(0)

                        # cls2_preds = []
                # print(cls2_labels)
                # print(cls2_preds)
                # pdb.set_trace()

            # Log info
            progress_bar.update(batch_size)
            # break
            # save predictions for qualititative analysis
        # pdb.set_trace()
        # np.save(os.path.join(args.save_path, 'logits_multis1.npy'),np.array(logits_multis))
        # np.save(os.path.join(args.save_path, 'targets1.npy'),np.array(targets))
        # np.save('logits_multis.npy',np.array(logits_multis))
        # np.save('targets.npy',np.array(targets))




        ##================= real/fake cls ========================##
        # pdb.set_trace()
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        AUC_cls = roc_auc_score(y_true, y_pred)
        ACC_cls = cls_acc_all / cls_nums_all
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
        EER_cls = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        ##================= multi-label cls ========================##
        MAP = multi_label_meter.value().mean()
        OP, OR, OF1, CP, CR, CF1 = multi_label_meter.overall()
        # pdb.set_trace()

        log.info(f'Test Binary Acc: {ACC_cls}, AUC: {AUC_cls}, EER: {EER_cls}\n')
        log.info(f'Test Multicls mAP: {MAP.item()}, CF1: {CF1}, OF1: {OF1}\n')

        if args.stage == 'stage2':
            acc, f1, p, r, _ = evaluate_text(pred_list,target_list)
            EM = np.mean(EM)
            util.pred_to_csv(face_id_list,box_res,target_boxes)

            iou, iou50, iou75 = eval_iou_batch(torch.tensor(box_res).squeeze(dim=1), torch.tensor(target_boxes).squeeze(dim=1))
            log.info(f'Test IOU: {iou}, IOU 50: {iou50}, IOU 75: {iou75}\n')
            log.info(f'Test Acc: {acc}, F1: {f1}, EM: {EM}, Recall: {r}, Precision: {p}')
            # pdb.set_trace()
            # ap, ap50, ap75 = eval_ap_batch(box_res,target_boxes)
            # log.info(f'Test AP: {ap}, AP 50: {ap50}, AP 75: {ap75}\n')

        else:
            util.save_csv_preds(pred_list_all, args.res_dir)




def test_co(model,args):
    log = util.get_logger(args.record_dir, "root", "debug")
    util.set_seed(args.seed)
    # device, gpu_ids = util.get_available_devices()
    torch.cuda.set_device(int(args.device[-1]))
    torch.backends.cudnn.benchmark = True
    device = torch.cuda.current_device()

    tokenizer = model.tokenizer
    # model.t5.parallelize()
    model.to(device)

    train_loader, dev_loader, test_loader = get_dataloaders(args, tokenizer = tokenizer, batch_size=args.batch_size, num_train=args.num_train, num_val=args.num_val, num_workers=args.num_workers, SC=True)
        # get_dataloaders(args, tokenizer = tokenizer, batch_size=args.batch_size, num_train=args.num_train, num_val=args.num_val,
        #                 num_workers=args.num_workers)
    # test_loader = dev_loader    # for dev eval

    # reset in case we used the -1 flag for all
    num_train = len(train_loader.dataset)
    num_val = len(dev_loader.dataset)
    num_test = len(test_loader.dataset)
    total_steps = ( (num_train // args.batch_size) * args.epochs)     # num times that optim.step() will be called
    total_train = num_train * args.epochs
    print('data loaded, start testing')

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=args.adam_eps)#,weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_steps)
    save_path = args.save_path + args.exp_name +'.pth'
    skip_training = args.skip_training
    if skip_training:
        print("Skip training...")
        args.epochs = 1

    epoch = 0       # number of times we have passed through entire set of training examples
    step = 0        # number of total examples we have done (will be epoch * len(data_set) at end of each epoch)
    best_acc = 0
    best_f1 = 0

    torch.cuda.empty_cache()
    
    # from peft import PeftModel, PeftConfig
    # model = PeftModel.from_pretrained(model, args.lora_path)
    # model = model.merge_and_unload()

    log.info(f'Evaluating at step {step}...')
    model.eval()        # put model in eval mode

    # See how the model is doing with exact match on tokens
    pred_list_all = []                      # accumulate for saving; list; one list per epoch

    ###############
    # Test (and to save results)
    ###############    
    pred_list_all = []
    pred_list = []
    target_list = []
    EM = []
    box_res = []
    id_list = []
    target_boxes = []
    total_matches_no_eos_ct = 0
    total_matches_with_eos_ct = 0
    em_0, em_1 = 0, 0 
    gt_0, gt_1 = 0, 0 
    with torch.no_grad(), \
        tqdm(total=num_test) as progress_bar:
        for batch_num, batch in enumerate(test_loader):
            batch_size = len(batch["image_ids"])
            # pdb.set_trace()

            src_imgs,tgt_ids,img_out,logits_txt, logits_bi = forward_test(model,device,batch)
            # pdb.set_trace()
            orig_text_output = batch["target_text"]
            if args.model_type != 't5':
                tgt_ids = tgt_ids.argmax(dim=-1)
                orig_text_output = [int(i) for i in tgt_ids]
            # pdb.set_trace()
            if args.stage == "stage2":
                pred, gts , batch_acc, batch_EM, em_0, em_1,gt_0, gt_1  = batch_post_process_cls(batch['target_text'], logits_txt, logits_bi,em_0, em_1,gt_0, gt_1 )
                EM.append(batch_EM)
                ids = [ids[28:-4] for ids in batch["image_ids"]]
                id_list.extend(ids)
                pred_list.extend(pred)
                target_list.extend(gts)

                for out, box, src_img, id in zip(img_out['pred_boxes'],batch['bboxes'], src_imgs, id_list):
                    mask = box > 0
                    length = int(len(box[mask])/4)
                    if length != 0:
                        h, w, _ = src_img.shape
                        max_len = max(h,w)
                        out = out[:length]
                        box = box[:length]#.cpu().detach().numpy()                  
                        box_res.append(bbox_cxcywh_to_xyxy(out*max_len).cpu().detach().numpy())
                        target_boxes.append(bbox_cxcywh_to_xyxy(box*max_len).cpu().detach().numpy())
                        print(bbox_cxcywh_to_xyxy(out*max_len).cpu().detach().numpy())
                        print(bbox_cxcywh_to_xyxy(box*max_len).cpu().detach().numpy())
                        pdb.set_trace()
                        

            # Log info
            progress_bar.update(batch_size)
            # break
            # save predictions for qualititative analysis
        # pdb.set_trace()
        # # a是我需要保存的数据，其数据类型为tensor，但是还没有转换为numpy格式。
        # np.save("target_list.npy",target_list)
        # np.save("pred_list.npy",pred_list)


        if args.stage == 'stage2':
            acc, f1, p, r, _ = evaluate_text(pred_list,target_list)
            EM = np.mean(EM)
            # util.pred_to_csv(id_list,box_res,target_boxes)
            ap, ap50, ap75 = eval_ap_batch(box_res, target_boxes)
            log.info(f'Test Acc: {acc}, F1: {f1}, EM: {EM}, Recall: {r}, Precision: {p}')
            log.info(f'Test AP: {ap}, AP 50: {ap50}, AP 75: {ap75}\n')
            log.info(f'Test EM0: {em_0}, EM1: {em_1}\n')
            # print(gt_0, gt_1)
            # pdb.set_trace()
        else:
            util.save_csv_preds(pred_list_all, args.res_dir)