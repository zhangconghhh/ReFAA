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
from ms_cross_attn1 import MSCrossAttnBlockDAUL1 as MSCrossAttnBlock


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
        
        self.cls_multi1 = nn.Sequential(nn.Linear(456*32, 2048, bias=True),nn.Linear(2048, 256, bias=True),nn.Linear(256, 2, bias=True))
        self.cls_multi2 = nn.Sequential(nn.Linear(456*32, 2048, bias=True),nn.Linear(2048, 256, bias=True),nn.Linear(256, 2, bias=True))
        self.cls_bi = nn.Sequential(nn.Linear(458*32, 2048, bias=True),nn.Linear(2048, 256, bias=True),nn.Linear(256, 2, bias=True))
        self.softmax = torch.nn.Softmax(dim=1)
        self.model_type = args.model_type       
        self.mmfuse_res = MSCrossAttnBlock(d_model=256)        
        self.proj_layer_v = nn.Linear(256, 768, bias=True)
        self.proj_layer_v1 = nn.Linear(256, 768, bias=True)        
        self.proj_layer_t = nn.Linear(768, 768, bias=True)       
        self.proj_layer_t1 = nn.Linear(768, 768, bias=True)
        self.proj_layer_m = nn.Linear(768, 768, bias=True)
        self.proj_layer_m1 = nn.Linear(768, 768, bias=True)
      

        self.bbox_logits_proj = nn.Linear(256, 3, bias=True)
        self.srm_conv0 = SRMConv2d_Separate(256, 256)
        self.srm_conv1 = SRMConv2d_Separate(256, 256)
        self.srm_conv2 = SRMConv2d_Separate(256, 256)
        self.srm_conv3 = SRMConv2d_Separate(256, 256)
        self.loss_txt = nn.CrossEntropyLoss(ignore_index=0, weight=torch.FloatTensor([1,1,2]))
       
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

    def split_memory(self, memory_visual_res, sls=[75, 38, 19, 10]):
        slss = [sls[0]*sls[0], sls[0]*sls[0]+sls[1]*sls[1], sls[0]*sls[0]+sls[1]*sls[1]+sls[2]*sls[2]]
        memory_visual_0 = memory_visual_res[:,:slss[0],:]#.reshape(bs, sls[0], sls[0], channel).permute(0, 3, 1, 2)
        memory_visual_1 = memory_visual_res[:,slss[0]:slss[1],:]#.reshape(bs, sls[1], sls[1], channel).permute(0, 3, 1, 2)
        memory_visual_2 = memory_visual_res[:,slss[1]:slss[2],:]#.reshape(bs, sls[2], sls[2], channel).permute(0, 3, 1, 2)
        memory_visual_3 = memory_visual_res[:,slss[2]:,:]#.reshape(bs, sls[3], sls[3], channel).permute(0, 3, 1, 2)
        return [memory_visual_0, memory_visual_1, memory_visual_2, memory_visual_3]
        # return [memory_visual_0, memory_visual_1, memory_visual_2]

    def forward_train_cls(self, text = None, image = None, input_ids = None,attention_mask = None, labels_cls=None, labels=None, return_dict=True, src_out = None, batch = None):
        # encode text
        if input_ids is None:
            input_ids = self.tokenizer.encode(text, return_tensors="pt")
            encoder_output = self.text_model.encoder(input_ids = input_ids.to(self.args.device))
       
        label_bi = torch.zeros(len(batch['image_ids']), dtype=torch.long)
        label_bi[np.array(batch['target_text'])=='Yes'] = 1
        # label_multi, label_bi, label_multi1, label_multi2 = self.get_multi_label(batch['fake_cls'])#.to(self.args.device)

        encoder_output = self.text_model.encoder(input_ids = input_ids, attention_mask = attention_mask)
        encoded_text_feature = encoder_output[0] # [8,512,768]
        memory, memory_text, memory_visual, src_flatten, memory_res = self.dino.to(self.args.device).feat_enhance_and_detr_enc(image.to(self.args.device), encoded_text_feature.to(self.args.device), src_out.to(self.args.device))
       
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
        # pdb.set_trace()
        memory_V, memory_V_res = self.mmfuse_res(self.split_memory(memory_visual), self.split_memory(memory_res))
        bs = memory_res.shape[0]
        # pdb.set_trace()


        memory_txt_cls =  torch.concat([self.proj_layer_t(memory_text_res), self.proj_layer_t1(encoded_text_feature)], dim=1)
      
        memory_multi   = torch.concat([memory_txt_cls, self.proj_layer_v( torch.cat([memory_V,memory_V_res,memory],dim=1)), ], dim=1)
        logits_bi = self.cls_bi(self.multi_cls_layer_bi(memory_multi).view(bs, -1))
        # loss_bi = cross_entropy(logits_bi, label_bi.to(self.args.device), weight=torch.FloatTensor([2,1]).to(self.args.device))
        loss_bi = cross_entropy(logits_bi, label_bi.to(self.args.device))#, weight=torch.FloatTensor([1,2]).to(self.args.device))

    #    torch.FloatTensor([1, 1])
        
        outputs['loss_bi'] = loss_bi
          
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
        if self.args.stage == "stage2":
            img_tgt_dict = self.dino(image.to(self.args.device), encoded_text_feature.to(self.args.device), src_out.to(self.args.device))
        else:
            img_tgt_dict = None
        memory_text0 = self.proj_layer(memory_text)
        memory_text = encoded_text_feature + memory_text0
        memory_text_res = memory_text0 - encoded_text_feature

        # text decoding with co-attended features
        encoder_output = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state = memory_text.to(self.args.device))
          
        # sequence_output = self.text_model.generate(encoder_outputs = encoder_output, attention_mask = attention_mask, labels = labels, temperature = 0.5, return_dict = return_dict, image_features = memory)
        # pdb.set_trace()       
        # logits_bi = self.tokenizer.batch_decode(sequence_output,skip_special_tokens=True)
          
        
        # _, sequence_output, hidden_states = self.text_model(decoder_input_ids = None, encoder_outputs = encoder_output, labels = labels, image_features = memory, return_dict = return_dict)        
        memory_V, memory_V_res = self.mmfuse_res(self.split_memory(memory_visual), self.split_memory(memory_res))
        bs = memory_res.shape[0]      
        memory_txt_cls =  torch.concat([self.proj_layer_t(memory_text_res), self.proj_layer_t1(encoded_text_feature)], dim=1)
        # memory_multi   = torch.concat([memory_txt_cls, self.proj_layer_v(torch.cat([memory_V,memory_V_res],dim=1))], dim=1) 
        memory_multi   = torch.concat([memory_txt_cls, self.proj_layer_v( torch.cat([memory_V,memory_V_res,memory],dim=1)), ], dim=1)
  
        logits_bi = self.cls_bi(self.multi_cls_layer_bi(memory_multi).view(bs, -1))
            
    

        return None, img_tgt_dict, None, logits_bi
 
    
 