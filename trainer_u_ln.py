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
                w = nn.Parameter(child.weight.detach().clone())                   
                b = nn.Parameter(child.bias.detach().clone())                    
                self.weight.append(w)
                self.bias.append(b)           
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


class MultiTaskLoader:
    def __init__(
        self,
        task_loaders: List[DataLoader],
        mode: str = "power",
        steps_per_epoch: Optional[int] = None,
        *,
        sizes: Optional[List[int]] = None,
        alpha: float = 0.4,
        ratio: Optional[List[int]] = None,
        reshuffle_each_epoch: bool = True,
        seed: int = 42,
    ):
        
        assert len(task_loaders) >= 2, "Must two tasks"
        self.task_loaders = task_loaders
        self.num_tasks = len(task_loaders)
        self.mode = mode
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.rng = np.random.default_rng(seed)

      
        self._iters = [iter(dl) for dl in self.task_loaders]
        if mode == "power":
            assert sizes is not None and len(sizes) == self.num_tasks, "power need sizes"
            sizes = np.asarray(sizes, dtype=np.float64)
            weights = np.power(sizes, float(alpha))
            self._p = (weights / weights.sum()).astype(np.float64)
        elif mode == "ratio":
            assert ratio is not None and len(ratio) == self.num_tasks, "ratio need ratio"
            self._base_schedule = []
            for tid, k in enumerate(ratio):
                self._base_schedule += [tid] * int(k)
            self._reshuffle = bool(reshuffle_each_epoch)
        else:
            raise ValueError("mode must be power or ratio")

        self._built = False
        self._epoch_steps_left = None
        self._epoch_schedule = None
        self._ptr = 0

    def _reset_sub_iter(self, task_id: int):
        self._iters[task_id] = iter(self.task_loaders[task_id])

    def _build_epoch(self):
        if self.steps_per_epoch is None:
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
        self._built = False
        self._build_epoch()
        return self

    def __next__(self):
        if not self._built:
            self._build_epoch()

        if self._epoch_steps_left <= 0:
            raise StopIteration

        if self.mode == "power":
            task_id = int(self.rng.choice(self.num_tasks, p=self._p))
        else:  # ratio
            task_id = self._epoch_schedule[self._ptr]
            self._ptr += 1

        try:
            batch = next(self._iters[task_id])
        except StopIteration:
            self._reset_sub_iter(task_id)
            batch = next(self._iters[task_id])

        self._epoch_steps_left -= 1
        return task_id, batch



def genParaDict(wrapped):
    
    decoder = wrapped.model.dino.transformer.decoder
    domainln_params = []
    seen = set()  
    for m in decoder.modules():
        if isinstance(m, nn.LayerNorm) or re.search(r"norm", m.__class__.__name__, re.I):
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
    domainln_ids = {id(p) for p in domainln_params}
    base_params = [p for p in wrapped.parameters() if id(p) not in domainln_ids]
    return base_params, domainln_params




class GDinoWithDomainLN(nn.Module):
    def __init__(self, base_gdino: nn.Module, num_domains: int):
        super().__init__()
        self.model = base_gdino
        self.num_domains = num_domains
        self.decoder_ref = self._get_decoder_ref()
        replace_ln_in_decoder(self.decoder_ref, num_domains)

    def _get_decoder_ref(self):
        return self.model.dino.transformer.decoder  

    def set_domain(self, task_id: int):
        for m in self.decoder_ref.modules():
            if isinstance(m, DomainLN):
                m.set_domain(task_id)

    def forward_task(self, task_id: int, *args, **kwargs):
        self.set_domain(task_id)
        return self.model.forward_train_cls(*args, **kwargs,task_id=task_id)

    def forward_cls(self,  *args, **kwargs):
        return self.model.forward_cls(*args, **kwargs)

    def load_from_state_dict(self, args):
        save_name = args.checkpoint
        assert save_name is not None
        print('loading GDinoWithDomainLN from:',save_name)
        ckpt = torch.load(save_name, map_location=args.device)
        self.load_state_dict(ckpt['model'],strict=False)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def save_state_dict(self, args, epoch):
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
    total_steps = ( (num_train // args.batch_size) * args.epochs)     # num times that optim.step() will be called
    total_train = num_train * args.epochs
    print('data loaded, start training')

    # 小:大 = 1:2
    steps_per_epoch = 50000
    dl_small = train_loader_sc
    dl_big = train_loader_df
    train_loader = MultiTaskLoader(task_loaders=[dl_small, dl_big], mode="ratio",
                     steps_per_epoch=steps_per_epoch, ratio=[2, 1], reshuffle_each_epoch=True,)
    num_train = steps_per_epoch
     
   
    log.info(f'device: {device}\n'
            f'total_steps: {total_steps}\n'
            f'total_train (num_t * epoch): {total_train}\n'
            f'learning rate: {args.lr}\n'
            f'machine: {socket.gethostname()}\n')

    skip_training = args.skip_training
    if skip_training:
        print("Skip training...")
        args.epochs = 1
    weighter = make_weighter("dwa", num_tasks=2, T=2.0)
    epoch, step = 0, 0    
    model.train()
    wrapped = GDinoWithDomainLN(model, num_domains=2).to(device)
    optimizer = AdamW(wrapped.parameters(), lr=args.lr, eps=args.adam_eps)    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    while epoch < args.epochs:
        torch.cuda.empty_cache()
        epoch += 1        
        running = torch.zeros(2)
        count = 0
        with torch.enable_grad(), tqdm(total=num_train*args.batch_size) as progress_bar:
            for batch_num, (task_id, batch) in enumerate(train_loader):
                batch_size = len(batch["image_ids"])         
                L_task, _ = forward_u_ln(wrapped, tokenizer, device, batch, task_id=task_id)
                losses = [torch.tensor(0., device=L_task.device), torch.tensor(0., device=L_task.device)]
                losses[task_id] = L_task
                loss, _ = weighter.combine_step_equal(losses)            
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                running[task_id] += float(L_task.detach())
                count += 1             
                nn.utils.clip_grad_norm_(wrapped.parameters(), 10)
                optimizer.step()                
                scheduler.step()              

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
    wrapped.eval()    
    pred_list, target_list, box_res, target_boxes, EM = [], [], [], [], []
   
    with torch.no_grad(), tqdm(total=num_test) as progress_bar:
        multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
        multi_label_meter.reset()

        for batch_num, batch in enumerate(test_loader):     
            batch_size = len(batch["image_ids"])
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
                    box_res.append(out.tolist())
                    target_boxes.append(box.tolist())   
            iou, iou50, iou75 =  eval_ap_batch(torch.tensor(box_res), torch.tensor(target_boxes))   
            progress_bar.update(batch_size)
    
        _, f1, p, r, _ = evaluate_text(pred_list,target_list)
        EM = np.mean(EM)
        util.pred_to_csv(face_id_list,box_res,target_boxes)

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
    train_loader, dev_loader, test_loader = get_dataloaders(args, tokenizer = tokenizer, batch_size=args.batch_size, num_train=args.num_train, num_val=args.num_val, num_workers=args.num_workers)

    # reset in case we used the -1 flag for all
    num_train = len(train_loader.dataset)
    num_test = len(test_loader.dataset)
    total_steps = ( (num_train // args.batch_size) * args.epochs) 
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
            batch_size = len(batch["image_ids"])
            logits,tgt_ids,img_out, logits_multi, logits_bi = forward_test_cls_ln(model, tokenizer,device,batch)             
            if args.model_type != 't5': # false
                tgt_ids = tgt_ids.argmax(dim=-1)

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
            target, _, _, _ = model.get_multi_label(batch['fake_cls'])
            multi_label_meter.add(logits_multi, target)
            logits_multis.extend(logits_multi.cpu().tolist())
            targets.extend(target.cpu().tolist())

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
                batch_EM = 0

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


