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

def trainer(model, args):

    log = util.get_logger(args.record_dir, "root", "debug")
    util.set_seed(args.seed)
    torch.cuda.set_device(int(args.device[-1]))
    torch.backends.cudnn.benchmark = True
    tokenizer = model.tokenizer
    device = torch.cuda.current_device()
    model = model.to(device)
    
    train_loader, dev_loader, test_loader = get_dataloaders(args, tokenizer = tokenizer, batch_size=args.batch_size, num_train=args.num_train, num_val=args.num_val, num_workers=args.num_workers)
    num_train = len(train_loader.dataset) # 208184
    num_val = len(dev_loader.dataset)     # 22126
    num_test = len(test_loader.dataset)   # 50705
    total_steps = ( (num_train // args.batch_size) * args.epochs)     # num times that optim.step() will be called
    total_train = num_train * args.epochs
    print('data loaded, start training')

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=args.adam_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    save_path = args.save_path + args.exp_name +'.pth'

   
    log.info(f'device: {device}\n'
            f'total_steps: {total_steps}\n'
            f'total_train (num_t * epoch): {total_train}\n'
            f'learning rate: {args.lr}\n'
            f'machine: {socket.gethostname()}\n')

    skip_training = args.skip_training
    if skip_training:
        print("Skip training...")
        args.epochs = 1

    epoch =  0      # number of times we have passed through entire set of training examples
    step = 0        # number of total examples we have done (will be epoch * len(data_set) at end of each epoch)
    while epoch < args.epochs:
        torch.cuda.empty_cache()
        epoch += 1
        model.train()

        with torch.enable_grad(), tqdm(total=num_train) as progress_bar:
            for batch_num, batch in enumerate(train_loader):
               
                batch_size = len(batch["image_ids"])
                loss, _ = forward(model, device, batch)
                
                # Backwardfor name, param in model.named_parameters()
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print("nan gradient found")
                        print("name:",name)
                        print("param:",param.grad)
                        raise SystemExit

                optimizer.zero_grad()

                loss.backward()
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
        
        Test_F = True
        ###############
        # Test (and to save results)
        ###############
        if Test_F:
            pred_list_all = []
            pred_list = []
            target_list = []
            EM = []
            box_res = []
            id_list = []
            target_boxes = []
            targets = []
            logits_multis = []
            y_pred = []
            y_true= []
            cls_nums_all, cls_acc_all= 0, 0

            with torch.no_grad(), tqdm(total=num_test) as progress_bar:
                multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
                multi_label_meter.reset()

                for batch_num, batch in enumerate(test_loader):
                    batch_size = len(batch["image_ids"])
                    logits,tgt_ids,img_out, logits_multi, logits_bi = forward_test_cls(model,device,batch)
                    if args.model_type != 't5':
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
                                out = out[:length]#.cpu().detach().numpy()
                                box = box[:length]#.cpu().detach().numpy()
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
            

def text_lable(tgt_ids, logits):
    cls_label = tgt_ids.view(-1)
    cls_label[cls_label == 1] = 0
    cls_label[cls_label == -100] = 0
    cls_label[cls_label == 332] = 1  # 'T', Real text
    cls_label[cls_label == 411] = 2  # 'O', Fake text
    gts = cls_label[cls_label != 0]-1
    pred = torch.zeros(gts.shape)
    pred[torch.argmax(logits.view(-1, 3)[cls_label != 0], dim=1) == 2]=1
    return gts, pred

def test(model,args):
    log = util.get_logger(args.record_dir, "root", "debug")
    util.set_seed(args.seed)
    torch.cuda.set_device(int(args.device[-1]))

    tokenizer = model.tokenizer
    model.cuda()
    device = torch.cuda.current_device()

    train_loader, dev_loader, test_loader = \
        get_dataloaders(args, tokenizer = tokenizer, batch_size=args.batch_size, num_train=args.num_train, num_val=args.num_val,
                        num_workers=args.num_workers)
   
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

  
    step = 0        # number of total examples we have done (will be epoch * len(data_set) at end of each epoch)
    torch.cuda.empty_cache()   
    log.info(f'Evaluating at step {step}...')
    model.eval()        # put model in eval mode
    

    # See how the model is doing with exact match on tokens
    pred_list_all = []                      # accumulate for saving; list; one list per epoch

    ###############
    # Test (and to save results)
    ###############
    pred_list_all, pred_list, target_list, EM, logits_multis = [], [], [], [], []
    box_res, id_list, target_boxes, targets, y_pred, y_true = [], [], [], [], [], []
    cls_nums_all, cls_acc_all= 0, 0   

    with torch.no_grad(), tqdm(total=num_test) as progress_bar:
        multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
        multi_label_meter.reset()

        for batch_num, batch in enumerate(test_loader):
            batch_size = len(batch["image_ids"])
            logits,tgt_ids,img_out, logits_multi, logits_bi = forward_test_cls(model,tokenizer,device,batch)
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
                        out = out[:length]#.cpu().detach().numpy()
                        box = box[:length]#.cpu().detach().numpy()
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
