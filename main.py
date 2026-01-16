import pdb, torch
# from model_v1_v1_st1 import CofiPara
from model_v1_v1_u import CofiPara
# from model_v1_v1_mmsd1 import CofiPara
# from model_v1_v1 import CofiPara
from models.util.slconfig import SLConfig
from trainer_u_ln_lora import trainer, test
# from trainer_u import trainer, test
# from trainer import trainer, test
model_config_path = './model_config.py'
# pdb.set_trace()
args = SLConfig.fromfile(model_config_path)
model = CofiPara(args)
# pdb.set_trace()
# model = torch.compile(model) 没有加速作用zc
trainer(model,args)
# test(model, args)  
# test(model, args)      # if test
# test_cofi(model, args)
