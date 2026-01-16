import torch, pdb
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from models.t5.t5_model import T5ForConditionalGeneration
from models.groundingdino.util.inference import load_model, load_image, predict, annotate
from models.groundingdino.models.GroundingDINO.fuse_modules import FeatureResizer
import torchvision.transforms.functional as FF

from transformers import (
    AdamW,
    T5Config,
    RobertaConfig,
    T5TokenizerFast,
    RobertaModel,
    RobertaTokenizerFast,
    get_linear_schedule_with_warmup)
from models.util.slconfig import SLConfig
from utils import util
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions,SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from tqdm import tqdm
import socket
from collections import OrderedDict
from typing import *
from utils.loadData import *
from srm_utils import SRMConv2d_Separate
# from ms_cross_attn1 import MSCrossAttnBlockDAUL1 as MSCrossAttnBlock
from ms_cross_attn1 import MSCrossAttnBlockDAUL1_nocross as MSCrossAttnBlock
# from ms_cross_attn import  MSCrossAttnBlock
# from ms_cross_attn import MSCrossAttnBlockCLSDAUL as MSCrossAttnBlock
# from ms_cross_attn import MSCrossAttnBlockCLSDAUL1 as MSCrossAttnBlock
# from ms_cross_attn import MSCrossAttnBlockDAUL_SRM as MSCrossAttnBlock
# from visualizer import get_local
class FocalLoss(nn.Module):
    def __init__(self, gamma=0.25, weight=None, alpha=[0.5, 0.3, 0.2]):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        # alpha = self.alpha[targets]
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  
        pt = torch.exp(-ce_loss)  
        focal_loss = (1 - pt) ** self.gamma * ce_loss #* alpha
        return focal_loss


class CofiPara(torch.nn.Module):
    def __init__(self, args, params = None, **kwargs) -> None:
        super(CofiPara, self).__init__()
        assert args.model_type in ['t5','roberta']
        self.params = params
        self.args = args
        self.save_path = args.save_path
        # init dino model
        dino_config = "models/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        ckpt = args.g_dino_ckpt_path
        self.dino = load_model(dino_config,ckpt)
        self.proj_layer = nn.Linear(256, 768, bias=True)
        self.text_cls_layer = nn.Linear(768, 3, bias=True) # added by zc for text classification
        self.img_cls_layer = nn.Linear(512, 2, bias=True)


        self.multi_cls_layer2 = nn.Sequential(
            # nn.Linear(512, 256, bias=True),
            nn.Linear(768, 256, bias=True),
            nn.Linear(256, 256, bias=True),
            nn.LayerNorm(256, eps=1e-06),
            nn.GELU(),
            nn.Linear(256, 128, bias=True),
            nn.Linear(128, 64, bias=True),
            nn.LayerNorm(64, eps=1e-06),
            nn.GELU(),
            nn.Linear(64, 32, bias=True),
            nn.Linear(32, 32, bias=True))
        self.multi_cls_layer1 = nn.Sequential(
            # nn.Linear(512, 256, bias=True),
            nn.Linear(768, 256, bias=True),
            nn.Linear(256, 256, bias=True),
            nn.LayerNorm(256, eps=1e-06),
            nn.GELU(),
            nn.Linear(256, 128, bias=True),
            nn.Linear(128, 64, bias=True),
            nn.LayerNorm(64, eps=1e-06),
            nn.GELU(),
            nn.Linear(64, 32, bias=True),
            nn.Linear(32, 32, bias=True))
        self.multi_cls_layer_bi = nn.Sequential(
            # nn.Linear(512, 256, bias=True),
            nn.Linear(768, 256, bias=True),
            nn.Linear(256, 256, bias=True),
            nn.LayerNorm(256, eps=1e-06),
            nn.GELU(),
            nn.Linear(256, 128, bias=True),
            nn.Linear(128, 64, bias=True),
            nn.LayerNorm(64, eps=1e-06),
            nn.GELU(),
            nn.Linear(64, 32, bias=True),
            nn.Linear(32, 32, bias=True))
        # # 328 456 768 
        # self.cls_multi = nn.Sequential(nn.Linear(100*32, 256, bias=True),nn.Linear(256, 4, bias=True))
        # self.cls_bi = nn.Sequential(nn.Linear(100*32, 256, bias=True),nn.Linear(256, 2, bias=True))
        # self.cls_multi = nn.Sequential(
        #     nn.Linear(100*32, 2048, bias=True),
        #     nn.Linear(2048, 256, bias=True),
        #     # nn.LayerNorm(256, eps=1e-06),
        #     # nn.GELU(),
        #     nn.Linear(256, 4, bias=True))
        # self.cls_bi = nn.Sequential(nn.Linear(100*32, 2048, bias=True),nn.Linear(2048, 256, bias=True),nn.Linear(256, 2, bias=True))
        self.cls_multi1 = nn.Sequential(nn.Linear(456*32, 2048, bias=True),nn.Linear(2048, 256, bias=True),nn.Linear(256, 2, bias=True))
        self.cls_multi2 = nn.Sequential(nn.Linear(456*32, 2048, bias=True),nn.Linear(2048, 256, bias=True),nn.Linear(256, 2, bias=True))
        self.cls_bi = nn.Sequential(nn.Linear(656*32, 2048, bias=True),nn.Linear(2048, 256, bias=True),nn.Linear(256, 2, bias=True))
        # 456， 978
        self.softmax = torch.nn.Softmax(dim=1)
        self.model_type = args.model_type
        # self.mmfuse = MSCrossAttnBlock(d_model=256)
        self.mmfuse_res = MSCrossAttnBlock(d_model=256)
        # self.mmfuse_res1 = MSCrossAttnBlock(d_model=256)
        # self.mmfuse_res2 = MSCrossAttnBlock(d_model=256)
        # self.proj_layer_v_res = nn.Linear(768, 768, bias=True)
        # self.proj_layer_v = nn.Linear(768, 768, bias=True)
        # self.proj_layer_v_res = nn.Linear(256, 768, bias=True)
        self.proj_layer_v = nn.Linear(256, 768, bias=True)
        self.proj_layer_v1 = nn.Linear(256, 768, bias=True)
        # self.proj_layer_v = nn.Linear(768, 768, bias=True)
        # self.proj_layer_v1 = nn.Linear(768, 768, bias=True)
        self.proj_layer_t = nn.Linear(768, 768, bias=True)
        # self.proj_layer_vt -
        self.proj_layer_t1 = nn.Linear(768, 768, bias=True)
        self.proj_layer_m = nn.Linear(768, 768, bias=True)
        self.proj_layer_m1 = nn.Linear(768, 768, bias=True)
        # self.proj_layer_m = nn.Linear(256, 768, bias=True)

        self.bbox_logits_proj = nn.Linear(256, 3, bias=True)

        self.srm_conv0 = SRMConv2d_Separate(256, 256)
        self.srm_conv1 = SRMConv2d_Separate(256, 256)
        self.srm_conv2 = SRMConv2d_Separate(256, 256)
        self.srm_conv3 = SRMConv2d_Separate(256, 256)

        # self.loss_txt = nn.CrossEntropyLoss(ignore_index=0)#, weight=torch.FloatTensor([1,1,2]))
        self.loss_txt = nn.CrossEntropyLoss(ignore_index=0, weight=torch.FloatTensor([1,1,2]))
        # self.loss_multi1 = nn.CrossEntropyLoss(weight=torch.FloatTensor([3,1,1]))
        # self.loss_multi2 = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,5.7,13]))

        # self.loss_multi1 = nn.functional.binary_cross_entropy_with_logits(pos_weight=torch.FloatTensor([0.3, 0.3]))
        # self.loss_multi2 = nn.functional.binary_cross_entropy_with_logits(pos_weight=torch.FloatTensor([5.7,13]))
     


        # init t5 model
        # if not using checkpoint, use pretrained t5
        if self.model_type == 't5':
            if not args.checkpoint:
                self.text_model = T5ForConditionalGeneration.from_pretrained(args.model_path)
            else:
                config = T5Config.from_json_file(args.model_path+'/config.json')
                self.text_model = T5ForConditionalGeneration(config=config)

            self.tokenizer = T5TokenizerFast.from_pretrained(args.model_path)

        if args.checkpoint is not None:
            print('Loading form checkpoint:',args.model_path)
            self.load_from_state_dict(args)


    def save_state_dict(self, args, epoch):
        # save all parameters
        save_name = self.save_path + args.exp_name + str(epoch) +'.pth'
        print('saving to:',save_name)
        torch.save({'model':self.state_dict()},
                        save_name)

    def load_from_state_dict(self, args):
        save_name = args.checkpoint
        assert save_name is not None
        print('loading from:',save_name)
        ckpt = torch.load(save_name, map_location=args.device)
        # ckpt["model"].pop('dino.transformer.tgt_embed.weight')  # not needed
        self.load_state_dict(ckpt['model'],strict=False)

    def get_multi_label(self, label):
        multi_label = torch.zeros([len(label), 4], dtype=torch.long)
        bi_label = torch.ones([len(label)], dtype=torch.long)
        # multi_label1 = torch.zeros([len(label),2], dtype=torch.long)
        # multi_label2 = torch.zeros([len(label),2], dtype=torch.long)
        # origin cls = [0, 0, 0, 0]
        real_label_pos = np.where(np.array(label) == 'orig')[0].tolist()
        bi_label[real_label_pos] = 0
        multi_label[real_label_pos,:] = torch.tensor([0, 0, 0, 0])
        # face_swap cls = [1, 0, 0, 0]
        pos = np.where(np.array(label) == 'face_swap')[0].tolist()
        multi_label[pos,:] = torch.tensor([1, 0, 0, 0])
        # multi_label1[pos,0] = 1
        # face_attribute cls = [0, 1, 0, 0]
        pos = np.where(np.array(label) == 'face_attribute')[0].tolist()
        multi_label[pos,:] = torch.tensor([0, 1, 0, 0])
        # multi_label1[pos] = 2
        # text_swap cls = [0, 0, 1, 0]
        pos = np.where(np.array(label) == 'text_swap')[0].tolist()
        multi_label[pos,:] = torch.tensor([0, 0, 1, 0])
        # multi_label2[pos] = 1
        # text_attribute cls = [0, 0, 0, 1]
        pos = np.where(np.array(label) == 'text_attribute')[0].tolist()
        multi_label[pos,:] = torch.tensor([0, 0, 0, 1])
        # multi_label2[pos] = 2
        #  face_swap&text_swap cls = [1, 0, 1, 0]
        pos = np.where(np.array(label) == 'face_swap&text_swap')[0].tolist()
        multi_label[pos,:] = torch.tensor([1, 0, 1, 0])
        # multi_label1[pos] = 1
        # multi_label2[pos] = 1
        #  face_swap&text_attribute cls = [1, 0, 0, 1]
        pos = np.where(np.array(label) == 'face_swap&text_attribute')[0].tolist()
        multi_label[pos,:] = torch.tensor([1, 0, 0, 1])
        # multi_label1[pos] = 1
        # multi_label2[pos] = 2
        #  face_attribute&text_swap cls = [0, 1, 1, 0]
        pos = np.where(np.array(label) == 'face_attribute&text_swap')[0].tolist()
        multi_label[pos,:] = torch.tensor([0, 1, 1, 0])
        # multi_label1[pos] = 2
        # multi_label2[pos] = 1
        #  face_attribute&text_attribute cls = [0, 1, 0, 1]
        pos = np.where(np.array(label) == 'face_attribute&text_attribute')[0].tolist()
        multi_label[pos,:] = torch.tensor([0, 1, 0, 1])
        # multi_label1[pos] = 2
        # multi_label2[pos] = 2
        multi_label1 = multi_label[:,:2]
        multi_label2 = multi_label[:,2:]

        return multi_label, bi_label, multi_label1.type(torch.float).to(self.args.device), multi_label2.type(torch.float).to(self.args.device)

    def split_memory(self, memory_visual_res, sls=[75, 38, 19, 10]):
        slss = [sls[0]*sls[0], sls[0]*sls[0]+sls[1]*sls[1], sls[0]*sls[0]+sls[1]*sls[1]+sls[2]*sls[2]]
        memory_visual_0 = memory_visual_res[:,:slss[0],:]#.reshape(bs, sls[0], sls[0], channel).permute(0, 3, 1, 2)
        memory_visual_1 = memory_visual_res[:,slss[0]:slss[1],:]#.reshape(bs, sls[1], sls[1], channel).permute(0, 3, 1, 2)
        memory_visual_2 = memory_visual_res[:,slss[1]:slss[2],:]#.reshape(bs, sls[2], sls[2], channel).permute(0, 3, 1, 2)
        memory_visual_3 = memory_visual_res[:,slss[2]:,:]#.reshape(bs, sls[3], sls[3], channel).permute(0, 3, 1, 2)
        return [memory_visual_0, memory_visual_1, memory_visual_2, memory_visual_3]
        # return [memory_visual_0, memory_visual_1, memory_visual_2]

    def split_memory_srm(self, memory_visual_res, sls=[75, 38, 19, 10]):
        slss = [sls[0]*sls[0], sls[0]*sls[0]+sls[1]*sls[1], sls[0]*sls[0]+sls[1]*sls[1]+sls[2]*sls[2]]
        bs, _, channel = memory_visual_res.shape
        # pdb.set_trace()
        memory_visual_0 = self.srm_conv0(memory_visual_res[:,:slss[0],:].reshape(bs, sls[0], sls[0], channel).permute(0, 3, 1, 2)).view(bs, channel,-1).permute(0, 2, 1)
        memory_visual_1 = self.srm_conv1(memory_visual_res[:,slss[0]:slss[1],:].reshape(bs, sls[1], sls[1], channel).permute(0, 3, 1, 2)).view(bs, channel,-1).permute(0, 2, 1)
        memory_visual_2 = self.srm_conv2(memory_visual_res[:,slss[1]:slss[2],:].reshape(bs, sls[2], sls[2], channel).permute(0, 3, 1, 2)).view(bs, channel,-1).permute(0, 2, 1)
        memory_visual_3 = self.srm_conv3(memory_visual_res[:,slss[2]:,:].reshape(bs, sls[3], sls[3], channel).permute(0, 3, 1, 2)).view(bs, channel,-1).permute(0, 2, 1)
        return [memory_visual_0, memory_visual_1, memory_visual_2, memory_visual_3]
        # return [memory_visual_0, memory_visual_1, memory_visual_2]

    def forward_train_cls(self, text = None, image = None, input_ids = None,attention_mask = None, labels_cls=None, labels=None, return_dict=True, src_out = None, batch = None):
        # encode text
        if input_ids is None:
            input_ids = self.tokenizer.encode(text, return_tensors="pt")
            encoder_output = self.text_model.encoder(input_ids = input_ids.to(self.args.device))
        label_multi, label_bi, label_multi1, label_multi2 = self.get_multi_label(batch['fake_cls'])#.to(self.args.device)

        encoder_output = self.text_model.encoder(input_ids = input_ids, attention_mask = attention_mask)
        encoded_text_feature = encoder_output[0] # [8,512,768]
        memory, memory_text, memory_visual, src_flatten, memory_res = self.dino.to(self.args.device).feat_enhance_and_detr_enc(image.to(self.args.device), encoded_text_feature.to(self.args.device), src_out.to(self.args.device))
        # pdb.set_trace()
        # bbox decoding with dino
        if self.args.stage == "stage2":
            img_tgt_dict = self.dino(image.to(self.args.device), encoded_text_feature.to(self.args.device), src_out.to(self.args.device)) # keys:['pred_logits', 'pred_boxes']
        else:
            img_tgt_dict = None

        memory_text0 = self.proj_layer(memory_text) # [8,512,768]
        memory_text = encoded_text_feature + memory_text0
        memory_text_res = memory_text0 - encoded_text_feature

        # text decoding with co-attended features
        encoder_output = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state = memory_text.to(self.args.device)) #  encoder_output['last_hidden_state'] [bs,128,768]->[bs,128,768]
        outputs, sequence_output, _ = self.text_model(decoder_input_ids = None, encoder_outputs = encoder_output, labels = labels, image_features = memory, return_dict = return_dict)

        memory_V, memory_V_res = self.mmfuse_res(self.split_memory(memory_visual), self.split_memory(memory_res))
        pdb.set_trace()
        # memory_V1, memory_V_res1 = self.mmfuse_res1(self.split_memory(memory_visual), self.split_memory(memory_res))
        # memory_V2, memory_V_res2 = self.mmfuse_res2(self.split_memory(memory_visual), self.split_memory(memory_res))
    #    
        # memory_V, memory_V_res = self.mmfuse_res(self.split_memory(memory_visual), self.split_memory(memory_res),self.split_memory_srm(src_flatten))
        # memory_V1, memory_V_res1, memory_V_srm = self.mmfuse_res1(self.split_memory(memory_visual), self.split_memory(memory_res),self.split_memory_srm(src_flatten))

        # memory_V, memory_V_res, memory_V_srm = self.mmfuse_res(self.split_memory(memory_visual), self.split_memory(memory_res), self.split_memory_srm(src_flatten))
        bs = memory_res.shape[0]

        memory_txt_cls =  torch.concat([self.proj_layer_t(memory_text_res), self.proj_layer_t1(encoded_text_feature)], dim=1)
        # [656, 768],128, 100, 100, 128, 
        # pdb.set_trace()
        # biweight = self.cls_bi[0].weight.view(2048, 656, 32)
        # # import numpy as np
        

        memory_multi   = torch.concat([memory_txt_cls, self.proj_layer_v(torch.cat([memory_V,memory_V_res],dim=1)), self.proj_layer_m1(sequence_output)], dim=1)
        memory_multi1  = torch.concat([memory_txt_cls, self.proj_layer_v1(torch.cat([memory_V,memory_V_res],dim=1))], dim=1)
        memory_multit_txt  = torch.concat([memory_txt_cls, self.proj_layer_m(sequence_output)], dim=1)    
    
        logits_bi = self.cls_bi(self.multi_cls_layer_bi(memory_multi).view(bs, -1))
        # pdb.set_trace()
        loss_bi = cross_entropy(logits_bi, label_bi.to(self.args.device))

        logits_multi1 = self.cls_multi1(self.multi_cls_layer1(memory_multi1).view(bs, -1))
        logits_multi2 = self.cls_multi2(self.multi_cls_layer2(memory_multit_txt).view(bs, -1))

        

        cls_label = torch.zeros(labels.shape).to(labels.device).long()
        cls_label[labels == 332] = 1  # 'T', Real text
        cls_label[labels == 411] = 2  # 'O', Fake text
        logits = self.text_cls_layer(sequence_output)  # [8, 200, 768] -> [8, 200, 3]
        outputs['loss_txt'] = self.loss_txt(logits.view(-1, 3), cls_label.view(-1)) # [1600,3] [1600]
        
        outputs['loss_bi'] = loss_bi
        img_tgt_dict['pred_logits'] = self.bbox_logits_proj(img_tgt_dict['pred_logits'])
       
       
        loss_multi1 = binary_cross_entropy_with_logits(logits_multi1, label_multi1, pos_weight=torch.FloatTensor([3, 3]).to(self.args.device))
        # loss_multi2 = binary_cross_entropy_with_logits(logits_multi2, label_multi2, pos_weight=torch.FloatTensor([13,5.7]).to(self.args.device))
        loss_multi2 = binary_cross_entropy_with_logits(logits_multi2, label_multi2, pos_weight=torch.FloatTensor([1, 1]).to(self.args.device))#,weight=torch.FloatTensor([2, 1]).to(self.args.device))
        # pos_weight=[4.56,5.39,17.15,16.31] ,weight=[3.58,3.03,9.28,1]

        loss_multi = loss_multi1 + loss_multi2

        outputs['loss_multi'] = loss_multi


        return outputs, img_tgt_dict

    def forward_train(self, text = None, image = None, input_ids = None,attention_mask = None, labels_cls=None, labels=None, return_dict=True, src_out = None, batch = None):
        # encode text
        if input_ids is None:
            input_ids = self.tokenizer.encode(text, return_tensors="pt")
            encoder_output = self.text_model.encoder(input_ids = input_ids.to(self.args.device))
      
        encoder_output = self.text_model.encoder(input_ids = input_ids, attention_mask = attention_mask)
        encoded_text_feature = encoder_output[0] # [8,512,768]
        memory, memory_text, memory_visual, src_flatten, memory_res = self.dino.to(self.args.device).feat_enhance_and_detr_enc(image.to(self.args.device), encoded_text_feature.to(self.args.device), src_out.to(self.args.device))
        # pdb.set_trace()
        # bbox decoding with dino
        if self.args.stage == "stage2":
            img_tgt_dict = self.dino(image.to(self.args.device), encoded_text_feature.to(self.args.device), src_out.to(self.args.device)) # keys:['pred_logits', 'pred_boxes']
        else:
            img_tgt_dict = None

        memory_text0 = self.proj_layer(memory_text) # [8,512,768]
        memory_text = encoded_text_feature + memory_text0
        # memory_text_res = memory_text0 - encoded_text_feature

        # text decoding with co-attended features
        encoder_output = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state = memory_text.to(self.args.device)) #  encoder_output['last_hidden_state'] [bs,128,768]->[bs,128,768]
        outputs = self.text_model(
                                decoder_input_ids = None,
                                encoder_outputs = encoder_output,
                                labels = labels,
                                image_features = memory,
                                return_dict = return_dict
                                )

        # pdb.set_trace()
        return outputs, img_tgt_dict

    # @get_local('memory_text0')
    def forward_cls(self, text = None, image = None, input_ids = None,attention_mask = None,labels=None, labels_cls=None, return_dict=True, src_out = None, batch=None):
        # encode text
        if input_ids is None:
            input_ids = self.tokenizer.encode(text, return_tensors="pt")
            encoder_output = self.text_model.encoder(input_ids = input_ids.to(self.args.device))

        encoder_output = self.text_model.encoder(input_ids = input_ids, attention_mask = attention_mask)
        encoded_text_feature = encoder_output[0]
        memory, memory_text, memory_visual, src_flatten, memory_res = self.dino.to(self.args.device).feat_enhance_and_detr_enc(image.to(self.args.device), encoded_text_feature.to(self.args.device), src_out.to(self.args.device))
        # np.save('memory_visual.npy',np.array(memory_visual.cpu()))
        # np.save('memory_visual.npy',np.array(memory_res.cpu()))
        # pdb.set_trace()# 2 3 6 11
        # bbox decoding
        if self.args.stage == "stage2":
            img_tgt_dict = self.dino(image.to(self.args.device), encoded_text_feature.to(self.args.device), src_out.to(self.args.device))
        else:
            img_tgt_dict = None
        memory_text0 = self.proj_layer(memory_text)
        memory_text = encoded_text_feature + memory_text0
        memory_text_res = memory_text0 - encoded_text_feature

        # text decoding with co-attended features
        encoder_output = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state = memory_text.to(self.args.device))
        _, sequence_output, hidden_states = self.text_model(decoder_input_ids = None, encoder_outputs = encoder_output, labels = labels, image_features = memory, return_dict = return_dict)        # _, hidden_stat = self.text_model(decoder_input_ids = None, encoder_outputs = encoder_output, labels = labels_cls, image_features = memory, return_dict = return_dict)
       
        # sequence_output = self.text_model.generate(encoder_outputs = encoder_output, attention_mask = attention_mask, labels = labels, temperature = 0.5, return_dict = return_dict, image_features = memory)
        # text_output = self.tokenizer.batch_decode(sequence_output,skip_special_tokens=True)
        # pdb.set_trace()
        logits = self.text_cls_layer(sequence_output)


        memory_V, memory_V_res = self.mmfuse_res(self.split_memory(memory_visual), self.split_memory(memory_res))
        np.save('vg_sc.npy',np.array(memory_V.cpu().detach().numpy())) 
        np.save('vg_scr.npy',np.array(memory_V_res.cpu().detach().numpy())) 
        pdb.set_trace()
        # memory_V1, memory_V_res1 = self.mmfuse_res1(self.split_memory(memory_visual), self.split_memory(memory_res))
        # memory_V2, memory_V_res2 = self.mmfuse_res2(self.split_memory(memory_visual), self.split_memory(memory_res))
        # memory_V, memory_V_res = self.mmfuse_res(self.split_memory(memory_visual), self.split_memory(memory_res),self.split_memory_srm(src_flatten))
        # memory_V1, memory_V_res1, memory_V_srm = self.mmfuse_res1(self.split_memory(memory_visual), self.split_memory(memory_res),self.split_memory_srm(src_flatten))

        
        # memory_V, memory_V_res, memory_V_srm = self.mmfuse_res(self.split_memory(memory_visual), self.split_memory(memory_res), self.split_memory_srm(src_flatten))


        # memory_V, memory_V_res = self.mmfuse_res(self.split_memory(memory_visual), self.split_memory(memory_res))
        bs = memory_res.shape[0]

       


        # # memory_multi = self.proj_layer_v(torch.cat([memory_V,memory_V_res],dim=1))
        # # memory_multi1 = self.proj_layer_v1(torch.cat([memory_V,memory_V_res],dim=1))
        memory_txt_cls =  torch.concat([self.proj_layer_t(memory_text_res), self.proj_layer_t1(encoded_text_feature)], dim=1)
        # , self.proj_layer_m1(sequence_output)

        
        memory_multi   = torch.concat([memory_txt_cls, self.proj_layer_v(torch.cat([memory_V,memory_V_res],dim=1)), self.proj_layer_m1(sequence_output)], dim=1)
        memory_multi1  = torch.concat([memory_txt_cls, self.proj_layer_v1(torch.cat([memory_V,memory_V_res],dim=1))], dim=1)
        memory_multit_txt  = torch.concat([memory_txt_cls, self.proj_layer_m(sequence_output)], dim=1)    
        # pdb.set_trace()
        # # np.save('biweight.npy',np.array(biweight.cpu().detach().numpy()))  
        # # # memory_multi   = torch.concat([memory_txt_cls, self.proj_layer_v(memory_V)], dim=1)
        # # # memory_multi1  = torch.concat([memory_txt_cls, self.proj_layer_v1(memory_V)], dim=1)
        # # # memory_multit_txt  = torch.concat([memory_txt_cls, self.proj_layer_m(memory_V)], dim=1)   
        # memory_multi   = torch.concat([memory_txt_cls, self.proj_layer_v(torch.cat([memory_V,memory_V_res],dim=1))], dim=1)
        # # memory_multi1  = torch.concat([memory_txt_cls, self.proj_layer_v1(torch.cat([memory_V,memory_V_res,memory_V_srm],dim=1))], dim=1)
        # memory_multit_txt  = torch.concat([memory_txt_cls, self.proj_layer_m(torch.cat([memory_V,memory_V_res],dim=1))], dim=1)   
    

     
  
      
        logits_bi = self.cls_bi(self.multi_cls_layer_bi(memory_multi).view(bs, -1))


        Wc =  self.cls_bi[0].weight.view(2048, 656, 32)
        A = self.multi_cls_layer_bi(memory_multi)
        heat = (Wc * A[0]).sum(dim=0)                            # [H, W]
        heat = heat.clamp(min=0)
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
        # np.save('biheat.npy',np.array(heat.cpu().detach().numpy())) 
        # pdb.set_trace()

        # logits_multi_txt = self.cls_multi_t(self.multi_cls_layer_t(memory_text).view(bs, -1))

        logits_multi1 = self.cls_multi1(self.multi_cls_layer1(memory_multi1).view(bs, -1))#*10
        logits_multi2 = self.cls_multi2(self.multi_cls_layer2(memory_multit_txt).view(bs, -1))
        
        # logits_multi1 = torch.argmax(logits_multi1, dim=1)
        # logits_multi2 = torch.argmax(logits_multi2, dim=1)
        # # pdb.set_trace()
        # logits_multi = torch.zeros((logits_multi1.shape[0], 4))
        # logits_multi[logits_multi1==1, 0] = 1
        # logits_multi[logits_multi1==2, 1] = 1
        # logits_multi[logits_multi2==1, 2] = 1
        # logits_multi[logits_multi2==2, 3] = 1
        logits_multi = torch.concat([logits_multi1, logits_multi2], dim=1)
         

        return logits, img_tgt_dict, logits_multi, logits_bi
 
    
           
    def forward(self, text = None, image = None, input_ids = None,attention_mask = None,labels=None, return_dict=True, src_out = None):
            # encode text
            if input_ids is None:
                input_ids = self.tokenizer.encode(text, return_tensors="pt")
                encoder_output = self.text_model.encoder(input_ids = input_ids.to(self.args.device))

            if self.model_type == 't5':
                encoder_output = self.text_model.encoder(input_ids = input_ids,
                                                            attention_mask = attention_mask)
            else:
                encoder_output = self.text_model(input_ids = input_ids,
                                                    attention_mask=attention_mask)
            encoded_text_feature = encoder_output[0]
            
            # srcs, masks, text_dict, memory, memory_text = self.dino.to(self.args.device).feat_enhance_and_detr_enc(image.to(self.args.device),
            #                                                                                         encoded_text_feature.to(self.args.device),
            #                                                                                         src_out.to(self.args.device))
            memory, memory_text, memory_visual, src_flatten, memory_res = self.dino.to(self.args.device).feat_enhance_and_detr_enc(image.to(self.args.device), encoded_text_feature.to(self.args.device), src_out.to(self.args.device))
   
            # bbox decoding
            if self.args.stage == "stage2":
                img_tgt_dict = self.dino(image.to(self.args.device),
                                        encoded_text_feature.to(self.args.device),
                                        src_out.to(self.args.device))
            else:
                img_tgt_dict = None

            memory_text0 = self.proj_layer(memory_text) 
            memory_text = encoded_text_feature + memory_text0
            memory_text_res = memory_text0 - encoded_text_feature

            memory_V, memory_V_res = self.mmfuse_res(self.split_memory(memory_visual), self.split_memory(memory_res))
            pdb.set_trace()

        
            # text decoding with co-attended features
            if self.model_type == 't5':        
                encoder_output = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state = memory_text.to(self.args.device))

                _, sequence_output, _ = self.text_model(decoder_input_ids = None, encoder_outputs = encoder_output, labels = labels, image_features = memory, return_dict = return_dict)
              
            else:
                logits = self.classifier(memory_text.to(self.args.device))

                outputs =  logits.argmax(dim=1)

            text_output = None
            # sequence_output = None
            logits_txt = self.text_cls_layer(sequence_output)  
            memory_txt_cls =  torch.concat([self.proj_layer_t(memory_text_res), self.proj_layer_t1(encoded_text_feature)], dim=1)
    
            memory_multi   = torch.concat([memory_txt_cls, self.proj_layer_v(torch.cat([memory_V,memory_V_res],dim=1)), self.proj_layer_m1(sequence_output)], dim=1)
  
    
            # memory_txt_cls =  torch.concat([self.proj_layer_t(memory_text_res), self.proj_layer_t1(encoded_text_feature)], dim=1)
            # memory_multit_txt  = torch.concat([memory_txt_cls, self.proj_layer_m(sequence_output)], dim=1)    
            logits_bi = self.cls_bi(self.multi_cls_layer_bi(memory_multi).view(memory.shape[0], -1))
            
            return text_output, img_tgt_dict, logits_txt, logits_bi
