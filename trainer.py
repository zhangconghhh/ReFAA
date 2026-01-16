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
# def get_param_dict(args, model: nn.Module):
#     param_dicts = [
#         {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
#         {
#             "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
#             "lr": args.lr_backbone,
#         }
#     ]
#     return param_dicts


def trainer(model, args):

    log = util.get_logger(args.record_dir, "root", "debug")
    util.set_seed(args.seed)
    torch.cuda.set_device(int(args.device[-1]))
    torch.backends.cudnn.benchmark = True
    tokenizer = model.tokenizer

    device = torch.cuda.current_device()
    model = model.to(device)
    # (device, dtype=torch.long)

    # # Let's define the LoraConfig    
    # encoderV_keys = ["qkv"]
    # # trainable params: 2,838,528 || all params: 485,005,456 || trainable%: 0.5852569213159532
    # decoderT_keys = ["wi_0", "wi_1", "wo"]  + ["EncDecAttention.q","EncDecAttention.k","EncDecAttention.v","EncDecAttention.o"] + \
    #     ["SelfAttention.q","SelfAttention.k","SelfAttention.v","SelfAttention.o"] + ["proj_q1", "proj_k1", "proj_v1", "proj_q2", "proj_k2", "proj_v2", "proj_o1", "proj_o2"]
    # # trainable params: 4,718,592 || all params: 475,831,184 || trainable%: 0.9916525353243767
    # cross_kyes = ["value_proj", "output_proj", "out_proj", "v_proj", "l_proj", "values_v_proj", "values_l_proj", "out_v_proj", "out_l_proj", "value_proj"]
    # # trainable params: 2,555,904 || all params: 484,722,832 || trainable%: 0.5272918524291837
    # config = LoraConfig(r=8, inference_mode=False, lora_alpha=32, lora_dropout=0.1, bias="none",  target_modules=encoderV_keys+decoderT_keys+cross_kyes)

    # # Get our peft model and print the number of trainable parameters
    # model = get_peft_model(model, config)
    # model.print_trainable_parameters()
    
                
    # for name, param in model.named_parameters():
    #     # if 'cls' in name or 'mmfuse_res' in name or 'proj_layer' in name or 'srm_conv' in name or 'transformer.encoder' in name:
    #     if 'cls' in name or 'mmfuse_res' in name or 'proj_layer' in name or 'srm_conv' in name:
    #         param.requires_grad = True
 


    train_loader, dev_loader, test_loader = get_dataloaders(args, tokenizer = tokenizer, batch_size=args.batch_size, num_train=args.num_train, num_val=args.num_val, num_workers=args.num_workers)
    # train_loader  = dev_loader#[:500]
    num_train = len(train_loader.dataset) # 208184
    num_val = len(dev_loader.dataset)     # 22126
    num_test = len(test_loader.dataset)   # 50705
    total_steps = ( (num_train // args.batch_size) * args.epochs)     # num times that optim.step() will be called
    total_train = num_train * args.epochs
    print('data loaded, start training')

    # pdb.set_trace()
    # param_dicts = get_param_dict(args, model)
    # optimizer = AdamW(param_dicts, lr=args.lr, eps=args.adam_eps, weight_decay=args.weight_decay)
    # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=args.adam_eps, weight_decay=args.weight_decay)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=args.adam_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    save_path = args.save_path + args.exp_name +'.pth'

    # backupcpde
    import shutil
    # shutil.copyfile('/media/disk/01drive/01congzhang/24forensics/CofiPara-master/model.py', os.path.join(args.save_path, 'model.py'))
    # shutil.copyfile('/media/disk/01drive/01congzhang/24forensics/CofiPara-master/trainer.py', os.path.join(args.save_path, 'trainer.py'))
    # shutil.copyfile('/media/disk/01drive/01congzhang/24forensics/CofiPara-master/models/groundingdino/models/GroundingDINO/groundingdino.py', os.path.join(args.save_path, 'groundingdino.py'))
    # shutil.copyfile('/media/disk/01drive/01congzhang/24forensics/CofiPara-master/models/groundingdino/models/GroundingDINO/transformer.py', os.path.join(args.save_path, 'transformer.py'))
    # shutil.copyfile('/media/disk/01drive/01congzhang/24forensics/CofiPara-master/utils/loadData.py', os.path.join(args.save_path, 'loadData.py'))
    # shutil.copyfile('/media/disk/01drive/01congzhang/24forensics/CofiPara-master/model_config.py', os.path.join(args.save_path, 'model_config.py'))
    # shutil.copyfile('/media/disk/01drive/01congzhang/24forensics/CofiPara-master/ms_cross_attn.py', os.path.join(args.save_path, 'ms_cross_attn.py'))
    # # pdb.set_trace()


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

    epoch =  0      # number of times we have passed through entire set of training examples
    step = 0        # number of total examples we have done (will be epoch * len(data_set) at end of each epoch)
    while epoch < args.epochs:
        torch.cuda.empty_cache()
        epoch += 1
        model.train()

        with torch.enable_grad(), tqdm(total=num_train) as progress_bar:
            for batch_num, batch in enumerate(train_loader):
                # pdb.set_trace()
                # if batch_num == 0:
                #     log.info(f'**Training sample**\t Source: {batch["source_text"][0]}\t Target: {batch["target_text"][0]}\n')
                batch_size = len(batch["image_ids"])
                # pdb.set_trace()
                # zc comments  For fengci
                # Stage 1: ['image_ids'] = 'xxxx.jpg'    ['target_text']  'Yes'
                # ['source_text'] = 'ikea cutting board with hole for the blood . clever thinking .'
                # Stage 2： image_ids, source_text, target_text='stuck in traffic'
                # bboxes.shape=[4,10,4], box_labels.shape=[4,10]
                # label: "[['there', 'O'], ['is', 'O'], ['nothing', 'O'], ['more', 'O'], ['superb', 'O'], ['to', 'O'], ['do', 'O'], ['on', 'O'], ['a', 'O'], ['saturday', 'O'], ['be', 'O'], ['stuck', 'B-S'], ['in', 'I-S'], ['traffic', 'I-S'], ['!', 'O'], ['#', 'O'], ['happydays', 'O']]"

                # for DeepFake
                # "source_text": 'Kiss me I m Irish a festivalgoer watches the parade in New York'
                # "target_text": 'T T T T T T T T T T T T T'
                # "fake_cls": 'orig'
                # pdb.set_trace()
                loss, _ = forward(model, device, batch)
                # loss, _ = forward_mmsd(model, device, batch)
                # loss_val =      # get the item since loss is a tensor

                # Backwardfor name, param in model.named_parameters()
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print("nan gradient found")
                        print("name:",name)
                        print("param:",param.grad)
                        raise SystemExit
                # torch.autograd.set_detect_anomaly(True)
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
        
        # model.save_pretrained(args.lora_path)
        # model = PeftModel.from_pretrained(model, args.lora_path)
        # # model = model.merge_and_unload()
        model.save_state_dict(args, epoch)
        # model.merge_and_unload().save_state_dict(args, epoch)
        
        Test_F = True
        Test_F = False
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
                # np.save(os.path.join(args.save_path, 'logits_multis1.npy'),np.array(logits_multis))
                # np.save(os.path.join(args.save_path, 'targets1.npy'),np.array(targets))


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
    # pdb.set_trace()

    with torch.no_grad(), tqdm(total=num_test) as progress_bar:
        multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
        multi_label_meter.reset()

        for batch_num, batch in enumerate(test_loader):
            # pdb.set_trace()
            batch_size = len(batch["image_ids"])
            logits,tgt_ids,img_out, logits_multi, logits_bi = forward_test_cls(model,tokenizer,device,batch)
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
            logits,tgt_ids,img_out, logits_multi, logits_bi = forward_test_cls(model,tokenizer,device,batch)
            # pdb.set_trace()
            orig_text_output = batch["target_text"]
            if args.model_type != 't5': # false
                tgt_ids = tgt_ids.argmax(dim=-1)
                orig_text_output = [int(i) for i in tgt_ids]
            # gts, pred = text_lable(tgt_ids, logits)
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
            # cls_label = target[:,-1].to(device) + target[:,-2].to(device)  # for st
            cls_label = target[:,0].to(device) + target[:,1].to(device) # for df
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




def test_s1(model,args):
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
    total_matches_no_eos_ct, total_matches_with_eos_ct = 0, 0

    with torch.no_grad(), tqdm(total=num_test) as progress_bar:
        multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
        multi_label_meter.reset()

        for batch_num, batch in enumerate(test_loader):
            batch_size = len(batch["image_ids"])
         
            # generated_ids,tgt_ids,img_out = forward_test(model,device,batch)
            # # pdb.set_trace()
            # orig_text_output = batch["target_text"]
            # if args.model_type != 't5':
            #     tgt_ids = tgt_ids.argmax(dim=-1)
            #     orig_text_output = [int(i) for i in tgt_ids]

            # # collect some stats
            # if args.model_type == 't5':
            #     total_matches_no_eos, total_matches_with_eos, _ = \
            #         util.masked_token_match(tgt_ids, generated_ids, return_indices=True)
            # else:
            #     total_matches_no_eos = int(sum(tgt_ids==generated_ids))
            #     total_matches_with_eos = total_matches_no_eos
            # total_matches_no_eos_ct += total_matches_no_eos
            # total_matches_with_eos_ct += total_matches_with_eos

            # # todo: this could break once skip_special_tokens is fixed                    
            # if args.model_type == 't5':
            #     outputs_decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # else:
            #     outputs_decoded = [int(i) for i in generated_ids]
            # text_id = [imgid[-22:-4] for imgid in batch['image_ids']]
            # preds = list(zip(text_id, orig_text_output, outputs_decoded))
            # pred_list_all.extend(preds)
            # progress_bar.update(batch_size)


            orig_text_output = batch["target_text"]
            text_id = [imgid[-22:-4] for imgid in batch['image_ids']]
            loss, _ = forward_s1(model, device, batch)
            # pdb.set_trace()
            preds = list(zip(text_id, orig_text_output,  [loss.cpu().numpy().tolist()]))
            pred_list_all.extend(preds)
            progress_bar.update(batch_size)
           
        util.save_csv_preds(pred_list_all, args.res_dir)


def test_cofi(model,args):
    log = util.get_logger(args.record_dir, "root", "debug")
    util.set_seed(args.seed)
    torch.cuda.set_device(int(args.device[-1]))
    tokenizer = model.tokenizer
    model.cuda()
    device = torch.cuda.current_device()

    train_loader, dev_loader, test_loader = get_dataloaders(args, tokenizer = tokenizer, batch_size=args.batch_size, num_train=args.num_train, num_val=args.num_val, num_workers=args.num_workers)
    # test_loader = dev_loader    # for dev eval
    torch.cuda.empty_cache()
    model.eval()        # put model in eval mode
    # pdb.set_trace()

    num_test = len(test_loader.dataset)
    pred_list = []
    target_list = []
    EM = []
    box_res = []
    id_list = []
    target_boxes = []

    with torch.no_grad(), tqdm(total=num_test) as progress_bar:
        for batch_num, batch in enumerate(test_loader):
            batch_size = len(batch["image_ids"])
            logits,tgt_ids,img_out, logits_multi, logits_bi = forward_test_cls(model,device,batch)

            cls_label = tgt_ids.view(-1)
            cls_label[cls_label == 1] = 0
            cls_label[cls_label == -100] = 0
            cls_label[cls_label == 332] = 1  # 'T', Real text
            cls_label[cls_label == 411] = 2  # 'O', Fake text
            gts = cls_label[cls_label != 0]-1
            pred = torch.zeros(gts.shape)
            pred[torch.argmax(logits.view(-1, 3)[cls_label != 0], dim=1) == 2]=1
            # pdb.set_trace()
            batch_EM =  sum( pred == gts.cpu()) / len(gts)

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
                    target_boxes.append(box.tolist())
            progress_bar.update(batch_size)
            # pdb.set_trace()

        acc, f1, p, r, _ = evaluate_text(pred_list,target_list)
        EM = np.mean(EM)
        util.pred_to_csv(face_id_list,box_res,target_boxes)

        # iou, iou50, iou75 = eval_iou_batch(torch.tensor(box_res).squeeze(dim=1), torch.tensor(target_boxes).squeeze(dim=1))
        # log.info(f'Test IOU: {iou}, IOU 50: {iou50}, IOU 75: {iou75}\n')
     
        # pdb.set_trace()
        # pdb.set_trace()
        ap, ap50, ap75 = eval_ap_batch(box_res,target_boxes)
        log.info(f'Test AP: {ap}, AP 50: {ap50}, AP 75: {ap75}\n')
        log.info(f'Test Acc: {acc}, F1: {f1}, EM: {EM}, Recall: {r}, Precision: {p}')
            # pdb.set_trace()



def calculate_em(list1, list2):
    assert len(list1) == len(list2)
    return sum(1 for x, y in zip(list1, list2) if x == y) / len(list1)


def test_mmsd(model,args):
    log = util.get_logger(args.record_dir, "root", "debug")
    util.set_seed(args.seed)
    torch.cuda.set_device(int(args.device[-1]))
    tokenizer = model.tokenizer
    model.cuda()
    device = torch.cuda.current_device()

    train_loader, dev_loader, test_loader = get_dataloaders(args, tokenizer = tokenizer, batch_size=args.batch_size, 
                num_train=args.num_train, num_val=args.num_val, num_workers=args.num_workers)
    num_test = len(test_loader.dataset)
    torch.cuda.empty_cache()
    model.eval()        # put model in eval mode
    
    pred_list = []
    target_list = []
    y_pred = []
    y_true= []
    cls_nums_all, cls_acc_all= 0, 0
  
    with torch.no_grad(), tqdm(total=num_test) as progress_bar:
        multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
        multi_label_meter.reset()

        for batch_num, batch in enumerate(test_loader):
            # pdb.set_trace()
            batch_size = len(batch["image_ids"])
            logits,tgt_ids,img_out, logits_multi, logits_bi = forward_test_cls_mmsd(model,tokenizer,device,batch)
            
            cls_label = torch.zeros(len(batch['target_text']), dtype=torch.long).to(device)
            real_label_pos = np.where(np.array(batch['target_text']) == 'Yes')[0].tolist()
            cls_label[real_label_pos] = 1
            y_true.extend(cls_label.cpu().flatten().tolist())
          
            # cls_pred = torch.zeros(len(batch['target_text']), dtype=torch.long).to(device)
            # pre_label_pos = np.where(np.array(logits_bi) == 'Yes')[0].tolist()
            # cls_pred[pre_label_pos] = 1
            # y_pred.extend(cls_pred.cpu().flatten().tolist())
            y_pred.extend(logits_bi.argmax(1).cpu().flatten().tolist())

            progress_bar.update(batch_size)
            # pdb.set_trace()
          
            
        # pdb.set_trace()
        y_true, y_pred = np.array(y_true), np.array(y_pred)    
        # preds, gts = pred_list,target_list        
        p = precision_score(y_true, y_pred, average='binary')
        r = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        acc = accuracy_score(y_true, y_pred)    

        # acc, f1, p, r, _ = evaluate_text(pred_list,target_list)        
        log.info(f'Test Acc: {acc}, Precision: {p}, Recall: {r}, F1: {f1}')


        from sklearn.metrics import confusion_matrix  

        cm = confusion_matrix(y_true, y_pred)
        TN, FP, FN, TP = cm.ravel()

        print("TP:", TP, "FP:", FP, "TN:", TN, "FN:", FN)
        pdb.set_trace()
        
       
        
      

