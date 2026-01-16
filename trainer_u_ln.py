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

import torch
import torch.nn as nn

class DomainLN(nn.Module):
    """
    任务专属 LayerNorm:
    - 每个任务一套 gamma/beta
    - 归一化仍是标准 LN（逐样本逐 token）
    """
    def __init__(self, child, normalized_shape, num_domains, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.child = child
        self.num_domains = num_domains
        self.active_domain = 0#1
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:            
            self.weight = nn.ParameterList([])
            self.bias = nn.ParameterList([])

            for i in range(num_domains):
                # if i == 0:
                #     # 域0: weight=1, bias=0
                #     w = nn.Parameter(torch.ones(normalized_shape))
                #     b = nn.Parameter(torch.zeros(normalized_shape))
                # else:
                #     # 域1: 用当 前LayerNorm默认初始化（即标准LN初始化）
                w = nn.Parameter(child.weight.detach().clone())                   
                b = nn.Parameter(child.bias.detach().clone())                    
                self.weight.append(w)
                self.bias.append(b)


            # pdb.set_trace()
            # self.weight = nn.ParameterList([
            #     nn.Parameter(torch.ones(normalized_shape)) for _ in range(num_domains)
            # ])
            # self.bias = nn.ParameterList([
            #     nn.Parameter(torch.zeros(normalized_shape)) for _ in range(num_domains)
            # ])
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def set_domain(self, idx: int):
        self.active_domain = int(idx)

    def forward(self, x, domain=None):
        idx = self.active_domain if domain is None else domain
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            return x_norm * self.weight[idx] + self.bias[idx]
        else:
            return x_norm


def replace_ln_in_decoder(module: nn.Module, num_domains: int):
    """
    递归替换 decoder 中的 LayerNorm → DomainLN
    """
    for name, child in list(module.named_children()):
        replace_ln_in_decoder(child, num_domains)
        if isinstance(child, nn.LayerNorm):
            new_ln = DomainLN(
                child,
                child.normalized_shape,
                num_domains=num_domains,
                eps=child.eps,
                elementwise_affine=child.elementwise_affine
            )
            setattr(module, name, new_ln)
    return module


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



def genParaDict(wrapped):
    # import torch
    # import torch.nn as nn
    # import re

    # 1) 先拿到 decoder（按你的实际路径）
    decoder = wrapped.model.dino.transformer.decoder

    # 2) 收集 decoder 里所有 LayerNorm 的参数（兼容 nn.Parameter 与 ParameterList）
    domainln_params = []
    seen = set()  # 用 id 去重，防止同一参数被加入两次
    for m in decoder.modules():
        if isinstance(m, nn.LayerNorm) or re.search(r"norm", m.__class__.__name__, re.I):
            # 兼容两种仿射参数存放方式：
            # a) m.weight / m.bias 是 nn.Parameter
            # b) m.weight / m.bias 是 nn.ParameterList（形如 weight.0 / weight.1）
            for name in ("weight", "bias"):
                if hasattr(m, name) and getattr(m, name) is not None:
                    obj = getattr(m, name)
                    if isinstance(obj, nn.Parameter):
                        if id(obj) not in seen:
                            domainln_params.append(obj)
                            seen.add(id(obj))
                    elif isinstance(obj, nn.ParameterList):
                        for p in obj:
                            if id(p) not in seen:
                                domainln_params.append(p)
                                seen.add(id(p))

    # 3) 其他参数放到 base 组（避免与 domainln_params 重复）
    domainln_ids = {id(p) for p in domainln_params}
    base_params = [p for p in wrapped.parameters() if id(p) not in domainln_ids]
    return base_params, domainln_params

    




class GDinoWithDomainLN(nn.Module):
    """
    - encoder 保持共享
    - decoder 内 LayerNorm → DomainLN
    - set_domain(task_id) 控制任务专属 γ/β
    """
    def __init__(self, base_gdino: nn.Module, num_domains: int):
        super().__init__()
        self.model = base_gdino
        self.num_domains = num_domains
        # 定位 decoder
        self.decoder_ref = self._get_decoder_ref()
        # 替换 decoder 内所有 LayerNorm
        replace_ln_in_decoder(self.decoder_ref, num_domains)

    def _get_decoder_ref(self):
        return self.model.dino.transformer.decoder
       

    def set_domain(self, task_id: int):
        for m in self.decoder_ref.modules():
            if isinstance(m, DomainLN):
                m.set_domain(task_id)

    def forward_task(self, task_id: int, *args, **kwargs):
        # print(task_id)
        self.set_domain(task_id)
        return self.model.forward_train_cls(*args, **kwargs,task_id=task_id)


    def forward_cls(self,  *args, **kwargs):
        # self.set_domain(task_id)
        return self.model.forward_cls(*args, **kwargs)

    def load_from_state_dict(self, args):
        save_name = args.checkpoint
        assert save_name is not None
        print('loading GDinoWithDomainLN from:',save_name)
        ckpt = torch.load(save_name, map_location=args.device)
        self.load_state_dict(ckpt['model'],strict=False)


        # self.model.forward_train_cls(*args, **kwargs,task_id=task_id)



    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    def save_state_dict(self, args, epoch):
        # save all parameters
        save_name = args.save_path + args.exp_name + str(epoch) +'.pth'
        print('saving to:',save_name)
        torch.save({'model':self.state_dict()},
                        save_name)



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
    # steps_per_epoch = 6000
    dl_small = train_loader_sc
    dl_big = train_loader_df
    train_loader = MultiTaskLoader(task_loaders=[dl_small, dl_big], mode="ratio",
                     steps_per_epoch=steps_per_epoch, ratio=[2, 1], reshuffle_each_epoch=True,)
    #                steps_per_epoch=steps_per_epoch, ratio=[1, 3], reshuffle_each_epoch=True,)


    # train_loader = MultiTaskLoader1(task_loaders=[dl_small, dl_big], mode="ratio", steps_per_epoch=steps_per_epoch
    #                 , ratio=[3, 1], reshuffle_each_epoch=True, min_ratio={1: 0.30})

    # train_loader = MultiTaskLoader(task_loaders=[dl_big, dl_small], mode="power",
    #                 steps_per_epoch=steps_per_epoch, sizes=[208184, 3546], alpha=0.4, seed=2025,)
    # train_loader = MultiTaskLoader(task_loaders=[dl_big, dl_small], mode="power", steps_per_epoch=steps_per_epoch,
    #                 sizes=[208184, 3546], alpha=0.35, min_ratio={1: 0.30})

    num_train = steps_per_epoch

    import shutil
    shutil.copyfile('/data1/zc/24forensics/CofiPara-master/trainer_u_ln.py', os.path.join(args.save_path, 'trainer_u_ln.py'))
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
    
    model.train()
    wrapped = GDinoWithDomainLN(model, num_domains=2).to(device)

    # optimizer = AdamW(filter(lambda p: p.requires_grad, wrapped.parameters()), lr=args.lr, eps=args.adam_eps)
    optimizer = AdamW(wrapped.parameters(), lr=args.lr, eps=args.adam_eps)#,weight_decay=0.0)

    # base_params, domainln_params = genParaDict(wrapped)
    # ln_lr_factor = 5.0
    # optimizer = torch.optim.AdamW([{"params": base_params, "lr": args.lr},
    #                                {"params": domainln_params, "lr": args.lr*ln_lr_factor,"weight_decay": 0.0}], eps=args.adam_eps)

   
    
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    while epoch < args.epochs:
        torch.cuda.empty_cache()
        epoch += 1
        
        running = torch.zeros(2)
        count = 0

        with torch.enable_grad(), tqdm(total=num_train*args.batch_size) as progress_bar:
            for batch_num, (task_id, batch) in enumerate(train_loader):
                batch_size = len(batch["image_ids"])     
                # loss, _ = forward_u_ln(wrapped, tokenizer, device, batch, task_id=task_id)

                L_task, _ = forward_u_ln(wrapped, tokenizer, device, batch, task_id=task_id)
                losses = [torch.tensor(0., device=L_task.device), torch.tensor(0., device=L_task.device)]
                losses[task_id] = L_task
                # loss, _ = weighter.combine_step(losses)
                loss, _ = weighter.combine_step_equal(losses)            
                
                # optimizer.zero_grad()
                optimizer.zero_grad(set_to_none=True)

                loss.backward()
                running[task_id] += float(L_task.detach())
                count += 1             

                # nn.utils.clip_grad_norm_(wrapped.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(wrapped.parameters(), 10)

              
              
                optimizer.step()                
                scheduler.step()        # don't need to pass step to scheduler
           

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,loss= loss.item())
                
        
        
        wrapped.save_state_dict(args, epoch)
        if isinstance(weighter, DWAWeighter):
            avg_losses = (running / max(1, count)).tolist()
            weighter.update_epoch(avg_losses)



def test_co(model,args):
    log = util.get_logger(args.record_dir, "root", "debug")
    util.set_seed(args.seed)
    torch.cuda.set_device(int(args.device[-1]))
    device = torch.cuda.current_device()

    tokenizer = model.tokenizer
    model.cuda()
    wrapped = GDinoWithDomainLN(model, num_domains=2).to(device)    
    
    wrapped.load_from_state_dict(args)
    wrapped.set_domain(0)
    # model.dino.transformer.decoder.layers[1].norm1.weight[0][:5]
    # pdb.set_trace()
    
  
    train_loader, dev_loader, test_loader = get_dataloaders(args, tokenizer = tokenizer, batch_size=args.batch_size, num_train=args.num_train, num_val=args.num_val, num_workers=args.num_workers, SC=True)
    test_loader = dev_loader
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

  
    torch.cuda.empty_cache()
    wrapped.eval()        # put model in eval mode
    

   
    pred_list = []
    target_list = []
    EM = []
    box_res = []
    target_boxes = []
    
   
    with torch.no_grad(), tqdm(total=num_test) as progress_bar:
        multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
        multi_label_meter.reset()

        for batch_num, batch in enumerate(test_loader):    
            pdb.set_trace()       
            batch_size = len(batch["image_ids"])
            # logits,tgt_ids,img_out, _, _ = forward_test_cls(model,device,batch) 
            logits,tgt_ids,img_out, _, _ = forward_test_cls_ln(wrapped, tokenizer,device,batch)             
                      
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
                    out = out[:1].cpu()
                    box = box[:1].cpu()
                    # box_res.append([out.tolist()[0]])
                    # target_boxes.append(box.tolist()[0])  
                    box_res.append(out.tolist())
                    target_boxes.append(box.tolist())   
            # pdb.set_trace()
            iou, iou50, iou75 =  eval_ap_batch(torch.tensor(box_res), torch.tensor(target_boxes))   
            progress_bar.update(batch_size)
         

    
        _, f1, p, r, _ = evaluate_text(pred_list,target_list)
        EM = np.mean(EM)
        util.pred_to_csv(face_id_list,box_res,target_boxes)

        # iou, iou50, iou75 = eval_iou_batch(torch.tensor(box_res).squeeze(dim=1), torch.tensor(target_boxes).squeeze(dim=1))
        iou, iou50, iou75 =  eval_ap_batch(torch.tensor(box_res), torch.tensor(target_boxes))
        log.info(f'Test AP: {iou}, AP 50: {iou50}, AP 75: {iou75}\n')
        log.info(f'Test F1: {f1}, EM: {EM}, Recall: {r}, Precision: {p}')
       

def test(model,args):
    log = util.get_logger(args.record_dir, "root", "debug")
    util.set_seed(args.seed)
    torch.cuda.set_device(int(args.device[-1]))

    tokenizer = model.tokenizer
    model.cuda()
    device = torch.cuda.current_device()
    wrapped = GDinoWithDomainLN(model, num_domains=2).to(device)
    wrapped.load_from_state_dict(args)
    wrapped.set_domain(1)
    # pdb.set_trace()
   


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

 
    torch.cuda.empty_cache()
    wrapped.eval()        # put model in eval mode
    

 
    ###############
    # Test (and to save results)
    ###############
    pred_list_all,pred_list,target_list = [],[],[]     
    box_res,id_list,EM,y_pred,y_true = [],[],[],[],[]    
    target_boxes,targets, logits_multis = [],[],[]    
    cls_nums_all, cls_acc_all= 0, 0

   
    with torch.no_grad(), tqdm(total=num_test) as progress_bar:
        multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
        multi_label_meter.reset()

        for batch_num, batch in enumerate(test_loader):
            # pdb.set_trace()
            batch_size = len(batch["image_ids"])
            # logits,tgt_ids,img_out, logits_multi, logits_bi = forward_test_cls(model,tokenizer,device,batch)

            logits,tgt_ids,img_out, logits_multi, logits_bi = forward_test_cls_ln(model, tokenizer,device,batch)             
        
            # pdb.set_trace()
            orig_text_output = batch["target_text"]
            if args.model_type != 't5': # false
                tgt_ids = tgt_ids.argmax(dim=-1)
                orig_text_output = [int(i) for i in tgt_ids]
            # gts, pred = text_lable(tgt_ids, logits)
         
            # cls for binary and multicls
            ##================= real/fake cls ========================##
            cls_label = torch.ones(len(batch['fake_cls']), dtype=torch.long).to(device)
            real_label_pos = np.where(np.array(batch['fake_cls']) == 'orig')[0].tolist()
            cls_label[real_label_pos] = 0

            y_pred.extend(F.softmax(logits_bi,dim=1)[:,1].cpu().flatten().tolist())
            y_true.extend(cls_label.cpu().flatten().tolist())

            pred_acc = logits_bi.argmax(1)
            cls_nums_all += cls_label.shape[0]
            cls_acc_all += torch.sum(pred_acc == cls_label).item()

            # ----- multi metrics -----
            # target, _= model.get_multi_label(batch['fake_cls'])
            target, _, _, _ = model.get_multi_label(batch['fake_cls'])
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
                        out = out[:length]
                        box = box[:length]
                        # pdb.set_trace()
                        box_res.append([out.tolist()[0]])
                        face_id_list.append(face_id)
                        target_boxes.append(box.tolist()[0])

            progress_bar.update(batch_size)
          
        ##================= real/fake cls ========================##
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
           

        else:
            util.save_csv_preds(pred_list_all, args.res_dir)


