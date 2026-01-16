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
from collections import OrderedDict
from typing import *
from utils.loadData import *
from ms_cross_attn import MSCrossAttnBlockDAUL1_nocross as MSCrossAttnBlock
from ms_cross_attn import SRMConv2d_Separate



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
        num_dim = 768
        hidden_dim = 256

        # init t5 model
        if self.model_type == 't5':
            if not args.checkpoint:
                self.text_model = T5ForConditionalGeneration.from_pretrained(args.model_path)
            else:
                config = T5Config.from_json_file(args.model_path+'/config.json')
                self.text_model = T5ForConditionalGeneration(config=config)

            self.tokenizer = T5TokenizerFast.from_pretrained(args.model_path)

        self.dino = load_model(dino_config,ckpt)
        self.proj_layer = nn.Linear(256, 768, bias=True)
        self.text_cls_layer = nn.Linear(768, 3, bias=True)
        self.img_cls_layer = nn.Linear(512, 2, bias=True)

        self.multi_cls_layer2 = self.make_mlp_head()
        self.multi_cls_layer2 = self.make_mlp_head()
        self.multi_cls_layer_bi = self.make_mlp_head()
           
        self.cls_multi1 = self.make_cls_head(456)
        self.cls_multi2 = self.make_cls_head(456)        
        self.cls_bi = self.make_cls_head(656) 
      
        self.softmax = torch.nn.Softmax(dim=1)
        self.model_type = args.model_type
        self.mmfuse_res = MSCrossAttnBlock(d_model=256)

        self.proj_layer_v = nn.Linear(hidden_dim, num_dim, bias=True)
        self.proj_layer_v1 = nn.Linear(hidden_dim, num_dim, bias=True)
        self.proj_layer_t = nn.Linear(num_dim, num_dim, bias=True)
        self.proj_layer_t1 = nn.Linear(num_dim, num_dim, bias=True)
        self.proj_layer_m = nn.Linear(num_dim, num_dim, bias=True)
        self.proj_layer_m1 = nn.Linear(num_dim, num_dim, bias=True)        
        self.bbox_logits_proj = nn.Linear(hidden_dim, 3, bias=True)

        self.srm_conv0 = SRMConv2d_Separate(hidden_dim, hidden_dim)
        self.srm_conv1 = SRMConv2d_Separate(hidden_dim, hidden_dim)
        self.srm_conv2 = SRMConv2d_Separate(hidden_dim, hidden_dim)
        self.srm_conv3 = SRMConv2d_Separate(hidden_dim, hidden_dim)

        self.loss_txt = nn.CrossEntropyLoss(ignore_index=0, weight=torch.FloatTensor([1,1,2])) 

        if args.checkpoint is not None:
            print('Loading form checkpoint:',args.model_path)
            self.load_from_state_dict(args)


    def make_mlp_head(self, in_dim=768, eps=1e-6):
        return nn.Sequential(
            nn.Linear(in_dim, 256, bias=True),
            nn.Linear(256, 256, bias=True),
            nn.LayerNorm(256, eps=eps),
            nn.GELU(),
            nn.Linear(256, 128, bias=True),
            nn.Linear(128, 64, bias=True),
            nn.LayerNorm(64, eps=eps),
            nn.GELU(),
            nn.Linear(64, 32, bias=True),
            nn.Linear(32, 32, bias=True),
        )

    def make_cls_head(self, in_dim=656):
        return nn.Sequential(
            nn.Linear(in_dim*32, 2048, bias=True),
            nn.Linear(2048, 256, bias=True),
            nn.Linear(256, 2, bias=True),)


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
        # origin cls = [0, 0, 0, 0]
        real_label_pos = np.where(np.array(label) == 'orig')[0].tolist()
        bi_label[real_label_pos] = 0
        multi_label[real_label_pos,:] = torch.tensor([0, 0, 0, 0])
        # face_swap cls = [1, 0, 0, 0]
        pos = np.where(np.array(label) == 'face_swap')[0].tolist()
        multi_label[pos,:] = torch.tensor([1, 0, 0, 0])
        # face_attribute cls = [0, 1, 0, 0]
        pos = np.where(np.array(label) == 'face_attribute')[0].tolist()
        multi_label[pos,:] = torch.tensor([0, 1, 0, 0])
        # text_swap cls = [0, 0, 1, 0]
        pos = np.where(np.array(label) == 'text_swap')[0].tolist()
        multi_label[pos,:] = torch.tensor([0, 0, 1, 0])
        # text_attribute cls = [0, 0, 0, 1]
        pos = np.where(np.array(label) == 'text_attribute')[0].tolist()
        multi_label[pos,:] = torch.tensor([0, 0, 0, 1])
        #  face_swap&text_swap cls = [1, 0, 1, 0]
        pos = np.where(np.array(label) == 'face_swap&text_swap')[0].tolist()
        multi_label[pos,:] = torch.tensor([1, 0, 1, 0])
        #  face_swap&text_attribute cls = [1, 0, 0, 1]
        pos = np.where(np.array(label) == 'face_swap&text_attribute')[0].tolist()
        multi_label[pos,:] = torch.tensor([1, 0, 0, 1])
        #  face_attribute&text_swap cls = [0, 1, 1, 0]
        pos = np.where(np.array(label) == 'face_attribute&text_swap')[0].tolist()
        multi_label[pos,:] = torch.tensor([0, 1, 1, 0])
        #  face_attribute&text_attribute cls = [0, 1, 0, 1]
        pos = np.where(np.array(label) == 'face_attribute&text_attribute')[0].tolist()
        multi_label[pos,:] = torch.tensor([0, 1, 0, 1])
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

    def forward_train_cls(self, text = None, image = None, input_ids = None,attention_mask = None, labels_cls=None, labels=None, return_dict=True, src_out = None, batch = None):
        # encode text
        if input_ids is None:
            input_ids = self.tokenizer.encode(text, return_tensors="pt")
            encoder_output = self.text_model.encoder(input_ids = input_ids.to(self.args.device))
        label_multi, label_bi, label_multi1, label_multi2 = self.get_multi_label(batch['fake_cls'])#.to(self.args.device)

        encoder_output = self.text_model.encoder(input_ids = input_ids, attention_mask = attention_mask)
        encoded_text_feature = encoder_output[0] # [8,512,768]
        memory, memory_text, memory_visual, src_flatten, memory_res = self.dino.to(self.args.device).feat_enhance_and_detr_enc(image.to(self.args.device), encoded_text_feature.to(self.args.device), src_out.to(self.args.device))
      
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
        bs = memory_res.shape[0]
        memory_txt_cls =  torch.concat([self.proj_layer_t(memory_text_res), self.proj_layer_t1(encoded_text_feature)], dim=1) 
        memory_multi   = torch.concat([memory_txt_cls, self.proj_layer_v(torch.cat([memory_V,memory_V_res],dim=1)), self.proj_layer_m1(sequence_output)], dim=1)
        memory_multi1  = torch.concat([memory_txt_cls, self.proj_layer_v1(torch.cat([memory_V,memory_V_res],dim=1))], dim=1)
        memory_multit_txt  = torch.concat([memory_txt_cls, self.proj_layer_m(sequence_output)], dim=1)    
    
        logits_bi = self.cls_bi(self.multi_cls_layer_bi(memory_multi).view(bs, -1))
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
        loss_multi2 = binary_cross_entropy_with_logits(logits_multi2, label_multi2, pos_weight=torch.FloatTensor([1, 1]).to(self.args.device))
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
        # bbox decoding with dino
        if self.args.stage == "stage2":
            img_tgt_dict = self.dino(image.to(self.args.device), encoded_text_feature.to(self.args.device), src_out.to(self.args.device)) # keys:['pred_logits', 'pred_boxes']
        else:
            img_tgt_dict = None

        memory_text0 = self.proj_layer(memory_text) # [8,512,768]
        memory_text = encoded_text_feature + memory_text0
      
        # text decoding with co-attended features
        encoder_output = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state = memory_text.to(self.args.device)) #  encoder_output['last_hidden_state'] [bs,128,768]->[bs,128,768]
        outputs = self.text_model(decoder_input_ids = None,
                                encoder_outputs = encoder_output,
                                labels = labels,
                                image_features = memory,
                                return_dict = return_dict
                                )

        # pdb.set_trace()
        return outputs, img_tgt_dict

   
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
        _, sequence_output, hidden_states = self.text_model(decoder_input_ids = None, encoder_outputs = encoder_output, labels = labels, image_features = memory, return_dict = return_dict) 
        logits = self.text_cls_layer(sequence_output)

        memory_V, memory_V_res = self.mmfuse_res(self.split_memory(memory_visual), self.split_memory(memory_res))     
        memory_txt_cls =  torch.concat([self.proj_layer_t(memory_text_res), self.proj_layer_t1(encoded_text_feature)], dim=1)    
        memory_multi   = torch.concat([memory_txt_cls, self.proj_layer_v(torch.cat([memory_V,memory_V_res],dim=1)), self.proj_layer_m1(sequence_output)], dim=1)
        memory_multi1  = torch.concat([memory_txt_cls, self.proj_layer_v1(torch.cat([memory_V,memory_V_res],dim=1))], dim=1)
        memory_multit_txt  = torch.concat([memory_txt_cls, self.proj_layer_m(sequence_output)], dim=1)        

        bs = memory_res.shape[0]      
        logits_bi = self.cls_bi(self.multi_cls_layer_bi(memory_multi).view(bs, -1))
        logits_multi1 = self.cls_multi1(self.multi_cls_layer1(memory_multi1).view(bs, -1))
        logits_multi2 = self.cls_multi2(self.multi_cls_layer2(memory_multit_txt).view(bs, -1))
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
        memory, memory_text, memory_visual, src_flatten, memory_res = self.dino.to(self.args.device).feat_enhance_and_detr_enc(image.to(self.args.device), encoded_text_feature.to(self.args.device), src_out.to(self.args.device))

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
        
    
        # text decoding with co-attended features
        if self.model_type == 't5':        
            encoder_output = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state = memory_text.to(self.args.device))

            _, sequence_output, _ = self.text_model(decoder_input_ids = None, encoder_outputs = encoder_output, labels = labels, image_features = memory, return_dict = return_dict)
            
        else:
            logits = self.classifier(memory_text.to(self.args.device))

            outputs =  logits.argmax(dim=1)

        text_output = None
        logits_txt = self.text_cls_layer(sequence_output)  
        memory_txt_cls =  torch.concat([self.proj_layer_t(memory_text_res), self.proj_layer_t1(encoded_text_feature)], dim=1)
        memory_multi   = torch.concat([memory_txt_cls, self.proj_layer_v(torch.cat([memory_V,memory_V_res],dim=1)), self.proj_layer_m1(sequence_output)], dim=1)

        logits_bi = self.cls_bi(self.multi_cls_layer_bi(memory_multi).view(memory.shape[0], -1))
        
        return text_output, img_tgt_dict, logits_txt, logits_bi
