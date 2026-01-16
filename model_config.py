from tabnanny import check


g_dino_ckpt_path = "/data1/zc/24forensics/dgm4CofiPara/groundingdino_swint_ogc.pth" # https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
checkpoint = None
checkpoint = 'checkpoint251/0223modelv1_tv/cofi_para_fi_5k1.pth'
checkpoint = 'checkpoint251/0223modelv1_tv/cofi_para_fi_5k5.pth'
# checkpoint = 'checkpoint2510/1003_ratio_dwa_bb1_1/cofi_para_fi_5k1.pth'
# checkpoint = 'checkpoint2510/1006_5w_dwa_ln2/cofi_para_fi_5k2.pth'
# checkpoint = 'checkpoint2510/1006_5w_un_ln2/cofi_para_fi_5k2.pth'
# # checkpoint = 'checkpoint2510/1011_5w_dwa_ln2_1/cofi_para_fi_5k1.pth'
# checkpoint = 'checkpoint/0920_df/cofi_para_fi_5k3.pth'
# checkpoint = '/data1/zc/24forensics/CofiPara-master/checkpoint2510/mmsd_1/cofi_para_fi_5k5.pth'
# checkpoint = '/data1/zc/24forensics/CofiPara-master/checkpoint2510/mmsd_1017_1/cofi_para_fi_5k5.pth'
# checkpoint = '/data1/zc/24forensics/CofiPara-master/checkpoint2510/mmsd_1019_2/cofi_para_fi_5k1.pth'

# checkpoint = '/data1/zc/24forensics/CofiPara-master/checkpoint259/1003_ratio_dwa_bb1/cofi_para_fi_5k1.pth'
checkpoint = '/data1/zc/24forensics/CofiPara-master/checkpoint2510/1006_5w_dwa_ln2/cofi_para_fi_5k2.pth' # !!!!!!!!!!
# # checkpoint = '/data1/zc/24forensics/CofiPara-master/checkpoint2510/1011_5w_dwa_ln2_1/cofi_para_fi_5k1.pth'
# # checkpoint = '/data1/zc/24forensics/CofiPara-master/checkpoint2510/1011_5w_dwa_ln2_2/cofi_para_fi_5k1.pth'
# # checkpoint = '/data1/zc/24forensics/CofiPara-master/checkpoint2510/1013_5w_dwa_ln2_1/cofi_para_fi_5k1.pth'
# checkpoint= '/data1/zc/24forensics/CofiPara-master/checkpoint2510/1207_noscale/cofi_para_fi_5k2.pth'
# checkpoint = '/data1/zc/24forensics/CofiPara-master/checkpoint2510/1227_dwa_ln2_lora/cofi_para_fi_5k2.pth'
# checkpoint = '/data1/zc/24forensics/CofiPara-master/checkpoint2510/0107_dwa_ln2_lora/cofi_para_fi_5k2.pth'
# checkpoint = '/data1/zc/24forensics/CofiPara-master/checkpoint2510/0107_dwa_ln2_lora/cofi_para_fi_5k2.pth'
model_path = "/data1/zc/models/flan-t5-base"

stage = "stage2" # stage1 for pre-training and stage2 for fine-tuning
device = "cuda:2"
# save_path = "./checkpoint2510/mmsd_1019_lora/"
save_path = "./checkpoint2510/0107_dwa_ln2_lora/"
exp_name = "cofi_para_fi_5k"
model_type = 't5'

record_dir = save_path 
res_dir = save_path 
skip_training = False
seed = 42
batch_size = 64
epochs = 5
num_train = -14
num_val = -1
num_workers = 16 # 16 or 8sj
lr = 1e-4
# lr = 5e-4
# lr = 1e-5
# lr = 5e-5

weight_decay = 1e-4 # added by zc
lr_backbone = 1e-5 # added by zc
adam_eps = 1e-8
warmup_steps = 0
max_grad_norm =  1.0
use_wandb = False
alpha = 0.8

max_src_len = 128 # 512
max_tgt_len = 200
num_heads = 1
# export PYTHONPATH="$PWD:$PWD/models:$PWD/models/groundingdino:$PYTHONPATH"