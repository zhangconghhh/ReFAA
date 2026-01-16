import math, pdb
import torch 
import numpy as np
import torch.nn as nn
from functools import partial
from torch.nn import functional as F
from srm_utils import SRMConv2d_Separate
from models.groundingdino.models.GroundingDINO.mmf_deform_attn import MSDeformAttn as MSDeformAttn


"""
Channel Attention and Spaitial Attention from    
Woo, S., Park, J., Lee, J.Y., & Kweon, I. CBAM: Convolutional Block Attention Module. ECCV2018.
"""

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SRMConv2d_Separate(nn.Module):
    def __init__(self, inc, outc, learnable=False):
        super(SRMConv2d_Separate, self).__init__()
        self.inc = inc
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable) #[384,1,5,5]
        self.out_conv = nn.Sequential(
            nn.Conv2d(3*inc, outc, 1, 1, 0, 1, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )
        self.ca = ChannelAttention(outc)
        for ly in self.out_conv.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''
        out = F.conv2d(x, self.kernel, stride=1, padding=2, groups=self.inc)
        out = self.truc(out)
        out = self.out_conv(out)
        return out
    
    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, -2, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]
        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1], [filter2], [filter3]]
        filters = np.array(filters)
        filters = np.tile(filters, (inc,1,1,1))
        filters = torch.FloatTensor(filters)    
        return filters
    

class MSCrossAttnBlockDAUL1(nn.Module):
    def __init__(self, d_model=256, n_levels=3, n_heads=16, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 dropout=0.1, init_values=0.):
        super().__init__()
        self.select_layer = [_ for _ in range(n_levels)] #1
        self.query_layer = -1 
        self.cross_attn = CrossSelfAtt(d_model=d_model)       
        self.cross_attn1 = CrossSelfAtt(d_model=d_model)
        self.cross_attn2 = CrossSelfAtt(d_model=d_model)
        self.cross_attn_res = CrossSelfAtt(d_model=d_model)
        self.cross_attn_res1 = CrossSelfAtt(d_model=d_model)
        self.cross_attn_res2 = CrossSelfAtt(d_model=d_model)           
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, CrossSelfAtt):
                for n in m.modules():
                    if isinstance(n, MSDeformAttn):
                        n._reset_parameters()

    def forward(self, srcs, srcs_res):
        # prepare input feat
        src_flatten, src_flatten_res = [], []
        spatial_shapes = []
        for lvl in self.select_layer: 
            src = srcs[lvl]
            _, hw, _ = src.shape
            e = int(math.sqrt(hw))
            spatial_shape = (e, e)
            spatial_shapes.append(spatial_shape)
            src_flatten.append(src)
            src_flatten_res.append(srcs_res[lvl])         
            
        feat = torch.cat(src_flatten, 1)
        feat_res = torch.cat(src_flatten_res, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat.device) # shape=[3,2]
        ls_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])) # [   0, 5625, 7069]

        # cross attn
        query = srcs[self.query_layer]          
        query_res = srcs_res[self.query_layer]       
        attn = self.cross_attn(query, feat_res[:,:ls_index[1],:], spatial_shapes[0].unsqueeze(0), ls_index[0])
        attn_res = self.cross_attn_res(query_res, feat[:,:ls_index[1],:], spatial_shapes[0].unsqueeze(0), ls_index[0])
        attn1 = self.cross_attn1(attn, feat_res[:,ls_index[1]:ls_index[2],:], spatial_shapes[1].unsqueeze(0), ls_index[0])
        attn_res1 = self.cross_attn_res1(attn_res, feat[:,ls_index[1]:ls_index[2],:], spatial_shapes[1].unsqueeze(0), ls_index[0])       
        attn2 = self.cross_attn2(attn1, feat_res[:,ls_index[2]:,:], spatial_shapes[2].unsqueeze(0), ls_index[0])
        attn_res2 = self.cross_attn_res2(attn_res1, feat[:,ls_index[2]:,:], spatial_shapes[2].unsqueeze(0), ls_index[0])
    
        tgt = query + attn + attn1 + attn2
        tgt_res = query_res + attn_res + attn_res1 + attn_res2       
        return tgt, tgt_res


class MSCrossAttnBlockDAUL1_nocross(nn.Module):
    def __init__(self, d_model=256, n_levels=3, n_heads=16, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 dropout=0.1, init_values=0.):
        super().__init__()
        self.select_layer =[2,1,0]       
        self.query_layer = -1 

        self.cross_attn = CrossSelfAtt(d_model=d_model)       
        self.cross_attn1 = CrossSelfAtt(d_model=d_model)
        self.cross_attn2 = CrossSelfAtt(d_model=d_model)
        self.cross_attn_res = CrossSelfAtt(d_model=d_model)
        self.cross_attn_res1 = CrossSelfAtt(d_model=d_model)
        self.cross_attn_res2 = CrossSelfAtt(d_model=d_model)     
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, CrossSelfAtt):
                for n in m.modules():
                    if isinstance(n, MSDeformAttn):
                        n._reset_parameters()

    def forward(self, srcs, srcs_res):
        # prepare input feat
        src_flatten, src_flatten_res = [], []
        spatial_shapes = []
        for lvl in self.select_layer: 
            src = srcs[lvl]
            _, hw, _ = src.shape
            e = int(math.sqrt(hw))
            spatial_shape = (e, e)
            spatial_shapes.append(spatial_shape)
            src_flatten.append(src)
            src_flatten_res.append(srcs_res[lvl])
            
        feat = torch.cat(src_flatten, 1)
        feat_res = torch.cat(src_flatten_res, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat.device) # shape=[3,2]
        ls_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])) # [   0, 5625, 7069]

        # cross attn
        query = srcs[self.query_layer]          
        query_res = srcs_res[self.query_layer]
        attn = self.cross_attn(query, feat_res[:,:ls_index[1],:], spatial_shapes[0].unsqueeze(0), ls_index[0])
        attn_res = self.cross_attn_res(query_res, feat[:,:ls_index[1],:], spatial_shapes[0].unsqueeze(0), ls_index[0])       
        attn1 = self.cross_attn1(attn, feat_res[:,ls_index[1]:ls_index[2],:], spatial_shapes[1].unsqueeze(0), ls_index[0])
        attn_res1 = self.cross_attn_res1(attn_res, feat[:,ls_index[1]:ls_index[2],:], spatial_shapes[1].unsqueeze(0), ls_index[0])        
        attn2 = self.cross_attn2(attn1, feat_res[:,ls_index[2]:,:], spatial_shapes[2].unsqueeze(0), ls_index[0])
        attn_res2 = self.cross_attn_res2(attn_res1, feat[:,ls_index[2]:,:], spatial_shapes[2].unsqueeze(0), ls_index[0])
    
        tgt = query + attn + attn1 + attn2
        tgt_res = query_res + attn_res + attn_res1 + attn_res2       
        return tgt, tgt_res

class CrossSelfAtt(nn.Module):
    def __init__(self, d_model=768, n_levels=3, n_heads=16, n_points=4, init_values=0., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.cross_attn = MSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
        self.query_norm =  nn.Sequential(nn.Linear(d_model, d_model),norm_layer(d_model))
        self.feat_norm =  nn.Sequential(nn.Linear(d_model, d_model),norm_layer(d_model))
        self.self_attn = MSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
        self.norm = norm_layer(d_model)
        self.gamma1 = nn.Parameter(init_values * torch.ones((d_model)), requires_grad=True)
        self.gamma2 = nn.Parameter(init_values * torch.ones((d_model)), requires_grad=True)


    @staticmethod
    def get_reference_points(spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points
    
    
    def forward(self, query, feat, spatial_shapes, level_start_index):
        query_e = int(math.sqrt(query.shape[1])) 
        reference_points =  self.get_reference_points([(query_e, query_e)], device=query.device)
        attn = self.cross_attn(query=self.query_norm(query), reference_points=reference_points, input_flatten=self.feat_norm(feat), input_spatial_shapes=spatial_shapes, input_level_start_index=level_start_index, input_padding_mask=None) 
        attn1 = self.norm(attn)        
        query_e = int(math.sqrt(attn.shape[1]))
        spatial_shapes_attn = torch.as_tensor([(query_e, query_e)], dtype=torch.long, device=attn1.device)
        level_start_index_attn = torch.cat((spatial_shapes_attn.new_zeros((1,)), spatial_shapes_attn.prod(1).cumsum(0)[:-1]))
        reference_points_attn = self.get_reference_points(spatial_shapes_attn, device=attn1.device)
        attn2 = self.self_attn(attn1, reference_points_attn, attn1, spatial_shapes_attn, level_start_index_attn, None)   
        return attn* self.gamma1+attn2*self.gamma1
    

class CrossCrossSelfAtt(nn.Module):
    def __init__(self, d_model=768, n_levels=3, n_heads=16, n_points=4, init_values=0., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.cross_attn = MSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
        self.query_norm =  nn.Sequential(nn.Linear(d_model, d_model),norm_layer(d_model))
        self.feat_norm =  nn.Sequential(nn.Linear(d_model, d_model),norm_layer(d_model))
        self.self_attn = MSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
        self.norm = norm_layer(d_model)
        self.gamma1 = nn.Parameter(init_values * torch.ones((d_model)), requires_grad=True)
        self.gamma2 = nn.Parameter(init_values * torch.ones((d_model)), requires_grad=True)


        self.cross_attn_1 = MSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
        self.query_norm_1 =  nn.Sequential(nn.Linear(d_model, d_model),norm_layer(d_model))
        self.feat_norm_1 =  nn.Sequential(nn.Linear(d_model, d_model),norm_layer(d_model))
        self.self_attn_1 = MSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
        self.norm_1 = norm_layer(d_model)
        self.gamma1_1 = nn.Parameter(init_values * torch.ones((d_model)), requires_grad=True)
        self.gamma2_1 = nn.Parameter(init_values * torch.ones((d_model)), requires_grad=True)
    

    @staticmethod
    def get_reference_points(spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points
    
    
    def forward(self, query, feat, feat_1 , spatial_shapes, level_start_index):
        query_e = int(math.sqrt(query.shape[1])) 
        reference_points =  self.get_reference_points([(query_e, query_e)], device=query.device)
        attn = self.cross_attn(query=self.query_norm(query), reference_points=reference_points, input_flatten=self.feat_norm(feat), input_spatial_shapes=spatial_shapes, input_level_start_index=level_start_index, input_padding_mask=None) 
        attn1 = self.norm(attn)        
        query_e = int(math.sqrt(attn.shape[1]))
        spatial_shapes_attn = torch.as_tensor([(query_e, query_e)], dtype=torch.long, device=attn1.device)
        level_start_index_attn = torch.cat((spatial_shapes_attn.new_zeros((1,)), spatial_shapes_attn.prod(1).cumsum(0)[:-1]))
        reference_points_attn = self.get_reference_points(spatial_shapes_attn, device=attn1.device)
        attn2 = self.self_attn(attn1, reference_points_attn, attn1, spatial_shapes_attn, level_start_index_attn, None)    

        attn_1 = self.cross_attn_1(query=self.query_norm_1(attn2), reference_points=reference_points, input_flatten=self.feat_norm_1(feat_1), input_spatial_shapes=spatial_shapes, input_level_start_index=level_start_index, input_padding_mask=None) 
        attn1_1 = self.norm_1(attn_1)    
        attn2_1 = self.self_attn_1(attn1_1, reference_points_attn, attn1, spatial_shapes_attn, level_start_index_attn, None)    


       
        return attn*self.gamma1 + attn2*self.gamma1 + attn_1*self.gamma1_1 + attn2_1*self.gamma1_1