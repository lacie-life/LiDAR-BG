import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .vfe_template import VFETemplate
import numpy as np
import os
import open3d as o3d

from torch.quantization import (
    QuantStub,
    DeQuantStub
)

class PFNLayer_1x1conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            # self.linear = nn.Linear(in_channels, out_channels, bias=False)
            # print(in_channels , out_channels )
            self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size = 1, bias = False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

        self.relu = nn.ReLU()
        self.part = 50000

    def forward(self, inputs):
        

        inputs = inputs.permute(0,2,1)

        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.conv1x1(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            
            x = self.conv1x1(inputs)
            
        torch.backends.cudnn.enabled = False
        
        x = self.norm(x) if self.use_norm else x # for 1 x 1 conv
        
        x = x.permute(0,2,1)

        torch.backends.cudnn.enabled = True
        
        x = self.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        
        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            # print(in_channels , out_channels )
            # self.linear = nn.Conv1d(in_channels, out_channels, kernel_size = 1, bias = False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
            
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        # self.relu = nn.ReLU()
        self.part = 50000

    def forward(self, inputs):
        

        # inputs = inputs.permute(1,0)
        
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
            # x = inputs.view(inputs.shape[0],inputs.shape[1]*inputs.shape[2],inputs.shape[3]).permute(0,2,1) # for 1 x 1 conv
            # x = self.linear(x).squeeze() # for 1 x 1 conv
            
        if self.use_norm:

            torch.backends.cudnn.enabled = False
            
            # x = self.norm(x.permute(1,0)) if self.use_norm else x # for 1 x 1 conv
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            
            # x = x.view(inputs.shape[1],inputs.shape[2],x.shape[1]) # for 1 x 1 conv
            
            torch.backends.cudnn.enabled = True
        
        # x = self.relu(x)
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        
        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated

class PFNLayer_batch2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            # print(in_channels , out_channels )
            # self.linear = nn.Conv1d(in_channels, out_channels, kernel_size = 1, bias = False)
            self.norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
            
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.relu = nn.ReLU()
        self.part = 50000

    def forward(self, inputs):
        

        # inputs = inputs.permute(1,0)
        
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
            # x = inputs.view(inputs.shape[0],inputs.shape[1]*inputs.shape[2],inputs.shape[3]).permute(0,2,1) # for 1 x 1 conv
            # x = self.linear(x).squeeze() # for 1 x 1 conv
            
        if self.use_norm:

            torch.backends.cudnn.enabled = False
            
            # x = self.norm(x.permute(1,0)) if self.use_norm else x # for 1 x 1 conv
            x = x.unsqueeze(0)
            x = self.norm(x.permute(1, 3, 2, 0)).squeeze().permute(0,2,1)
            # x = x.view(inputs.shape[1],inputs.shape[2],x.shape[1]) # for 1 x 1 conv
            
            torch.backends.cudnn.enabled = True
        
        x = self.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        
        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated

class PFNLayer_QAT(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            # print(in_channels , out_channels )
            # self.linear = nn.Conv1d(in_channels, out_channels, kernel_size = 1, bias = False)
            
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

            # self.norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
            
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool1d(kernel_size = 32, stride = 32)
        self.part = 50000
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, inputs):
        
        inputs = self.quant(inputs)
        # inputs = inputs.permute(1,0)
        
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
            # x = inputs.view(inputs.shape[0],inputs.shape[1]*inputs.shape[2],inputs.shape[3]).permute(0,2,1) # for 1 x 1 conv
            # x = self.linear(x).squeeze() # for 1 x 1 conv
            
        if self.use_norm:
            torch.backends.cudnn.enabled = False
            
            # x = self.norm(x.permute(1,0)) if self.use_norm else x # for 1 x 1 conv
            # x = self.norm(x.permute(0, 2, 1)) if self.use_norm else x
            # x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
            
            x = self.norm(x.permute(0, 2, 1).unsqueeze(-1))   # for BatchNorm2d
            # x = x.squeeze().permute(0, 2, 1) # for BatchNorm2d
            
            # x = x.view(inputs.shape[1],inputs.shape[2],x.shape[1]) # for 1 x 1 conv
            
            torch.backends.cudnn.enabled = True
            
        # x = F.relu(x)
        x = self.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        
        # x_max = self.maxpool(x)
        # x_max = x_max.permute(0,2,1)

        if self.last_vfe:
            x_max = self.dequant(x_max)
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            x_concatenated = self.dequant(x_concatenated)
            return x_concatenated


class PFNLayer_QAT_1d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            # print(in_channels , out_channels )
            # self.linear = nn.Conv1d(in_channels, out_channels, kernel_size = 1, bias = False)
            
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

            # self.norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
            
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool1d(kernel_size = 32, stride = 32)
        self.part = 50000
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, inputs):
        
        inputs = self.quant(inputs)
        # inputs = inputs.permute(1,0)
        
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
            # x = inputs.view(inputs.shape[0],inputs.shape[1]*inputs.shape[2],inputs.shape[3]).permute(0,2,1) # for 1 x 1 conv
            # x = self.linear(x).squeeze() # for 1 x 1 conv
            
            
        torch.backends.cudnn.enabled = False
        
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x

        # x = self.norm(x.permute(0, 2, 1).unsqueeze(-1)) if self.use_norm else x # for BatchNorm2d
        # x = x.squeeze().permute(0, 2, 1) # for BatchNorm2d

        torch.backends.cudnn.enabled = True
        
        # x = F.relu(x)
        x = self.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        
        # x_max = self.maxpool(x)
        # x_max = x_max.permute(0,2,1)

        if self.last_vfe:
            x_max = self.dequant(x_max)
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            x_concatenated = self.dequant(x_concatenated)
            return x_concatenated

class PFNLayer_backgrond(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear_background = nn.Linear(in_channels, out_channels, bias=False)
            # print(in_channels , out_channels )
            # self.linear = nn.Conv1d(in_channels, out_channels, kernel_size = 1, bias = False)
            self.norm_background = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
            
        else:
            self.linear_background = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        

        # inputs = inputs.permute(1,0)
        
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear_background(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear_background(inputs)
            # x = inputs.view(inputs.shape[0],inputs.shape[1]*inputs.shape[2],inputs.shape[3]).permute(0,2,1) # for 1 x 1 conv
            # x = self.linear(x).squeeze() # for 1 x 1 conv
            
            
        torch.backends.cudnn.enabled = False
        
        # x = self.norm(x.permute(1,0)) if self.use_norm else x # for 1 x 1 conv
        x = self.norm_background(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        
        # x = x.view(inputs.shape[1],inputs.shape[2],x.shape[1]) # for 1 x 1 conv
        
        torch.backends.cudnn.enabled = True
        
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        
        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated
        
class PillarVFE_no_RELU(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer_no_RELU(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0] # 0.16
        self.voxel_y = voxel_size[1] # 0.16
        self.voxel_z = voxel_size[2] # 4
        
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #  -> 0.08 + 0
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1] #  -> 0.08 + -80 
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2] #  -> -1
        
        
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0): # max_num : 32
        
        """
        Create boolean mask by actually number of a padded tensor.

            Args:
                actual_num ([type]): [description] -> actual number of points per voxel
                max_num ([type]): [description] -> 

            Returns:
                [type]: [description]  
        """    
        
        actual_num = torch.unsqueeze(actual_num, axis + 1) 
        # tiled_actual_num: [N, M, 1]
        
        max_num_shape = [1] * len(actual_num.shape) # [1, 1]
       
        max_num_shape[axis + 1] = -1 # [1, -1]

        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        
        
        # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
        # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
        
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]

        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # print(voxel_num_points.shape)
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean 
        
        f_center = torch.zeros_like(voxel_features[:, :, :3]) 
        
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
            
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
            
        features = torch.cat(features, dim=-1)
        
        voxel_count = features.shape[1]
        
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        
        features *= mask 
        
        for pfn in self.pfn_layers:
            features = pfn(features)
            
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict
    
class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0] # 0.16
        self.voxel_y = voxel_size[1] # 0.16
        self.voxel_z = voxel_size[2] # 4
        
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #  -> 0.08 + 0
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1] #  -> 0.08 + -80 
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2] #  -> -1
        
        
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0): # max_num : 32
        
        """
        Create boolean mask by actually number of a padded tensor.

            Args:
                actual_num ([type]): [description] -> actual number of points per voxel
                max_num ([type]): [description] -> 

            Returns:
                [type]: [description]  
        """    
        
        actual_num = torch.unsqueeze(actual_num, axis + 1) 
        # tiled_actual_num: [N, M, 1]
        
        max_num_shape = [1] * len(actual_num.shape) # [1, 1]
       
        max_num_shape[axis + 1] = -1 # [1, -1]

        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        
        
        # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
        # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
        
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]

        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # print(voxel_num_points.shape)
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean 
        
        f_center = torch.zeros_like(voxel_features[:, :, :3]) 
        
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
            
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
            
        features = torch.cat(features, dim=-1)
        
        voxel_count = features.shape[1]
        
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        
        features *= mask 
        ### padding for onnx ###
        
        # if features.shape[0] <= 10000:
        #     padding = torch.zeros([10000 - features.shape[0],features.shape[1],features.shape[2]],dtype = torch.float32).to("cuda:0")
        #     features_padding = torch.cat([features,padding],dim = 0)

        #     # padding_coords = torch.zeros([10000-coords.shape[0], 4],dtype = torch.int32).to("cuda:0")
        #     # coords_padding = torch.cat([coords,padding_coords],dim=0)
        
        # features_raw = features_padding.detach().cpu().numpy().astype(np.float32)
        # features_raw.tofile(os.path.join("/mnt/hdd/snpe_228/pointpillar_6490_a9_v2/pillar_train_pfe_ground_points", "pillar_" + batch_dict["frame_id"][0] +".raw"))

        # voxel_coords_raw = coords_padding.detach().cpu().numpy().astype(np.float32)
        # voxel_coords_raw.tofile(os.path.join("/mnt/hdd/snpe_228/pointpillar_a9_v2/voxel_coords_train", "voxel_coords_" + batch_dict["frame_id"][0] +".raw"))
        
        # # print(os.path.join("/mnt/hdd/KITTI_dataset/pointpillar_preprocessing_python", batch_dict["frame_id"][0] +".raw"))
        # features_raw.tofile(os.path.join("/mnt/hdd/snpe_228/pointpillar_a9_v2/pillar_test", "pillar_" + batch_dict["frame_id"][0] +".raw"))
        
        # f = open("./mini_pointnet_train_list_100%.txt",'a')
        # f.write(os.path.join("/workspace/pointpillar_6490/mini_pointnet_train", "pillar_"+batch_dict["frame_id"][0]+".raw"))
        # f.write("\n")
        
        # voxel_coords_raw.tofile(os.path.join("/home/jiwon/3d_object_detection/CUDA-PointPillars/OpenPCDet/pointpillar_preprocessing_python", "voxel_coords_" + batch_dict["frame_id"][0] +".raw"))
        
        # # return 0
        # ### padding for onnx ###
        
        # # features = np.fromfile(os.path.join("/mnt/hdd/KITTI_dataset/pointpillar_preprocessing_c++","pillar_"+batch_dict["frame_id"][0] + ".bin"),dtype = np.float32).reshape(10000,32,10)
        # # features = torch.tensor(features).to("cuda:0")
        
        # # voxel_coords = np.fromfile(os.path.join("/mnt/hdd/KITTI_dataset/pointpillar_preprocessing_c++","voxel_coords_"+batch_dict["frame_id"][0] + ".bin"),dtype = np.int32).reshape(10000,4)
        # # voxel_coords = torch.tensor(voxel_coords).to("cuda:0")
        
        # # batch_dict['voxel_coords'] = voxel_coords
        
        # if features.shape[0] >= 30000:
        #     print(features.shape)
            
        for pfn in self.pfn_layers:
            features = pfn(features)
            
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict

class PillarVFE_batch2d(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer_batch2d(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0] # 0.16
        self.voxel_y = voxel_size[1] # 0.16
        self.voxel_z = voxel_size[2] # 4
        
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #  -> 0.08 + 0
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1] #  -> 0.08 + -80 
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2] #  -> -1
        
        
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0): # max_num : 32
        
        """
        Create boolean mask by actually number of a padded tensor.

            Args:
                actual_num ([type]): [description] -> actual number of points per voxel
                max_num ([type]): [description] -> 

            Returns:
                [type]: [description]  
        """    
        
        actual_num = torch.unsqueeze(actual_num, axis + 1) 
        # tiled_actual_num: [N, M, 1]
        
        max_num_shape = [1] * len(actual_num.shape) # [1, 1]
       
        max_num_shape[axis + 1] = -1 # [1, -1]

        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        
        
        # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
        # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
        
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]

        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # print(voxel_num_points.shape)
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean 
        
        f_center = torch.zeros_like(voxel_features[:, :, :3]) 
        
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
            
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
            
        features = torch.cat(features, dim=-1)
        
        voxel_count = features.shape[1]
        
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        
        features *= mask 
        ### padding for onnx ###
        
        # if features.shape[0] <= 10000:
        #     padding = torch.zeros([10000 - features.shape[0],features.shape[1],features.shape[2]],dtype = torch.float32).to("cuda:0")
        #     features_padding = torch.cat([features,padding],dim = 0)

        #     # padding_coords = torch.zeros([10000-coords.shape[0], 4],dtype = torch.int32).to("cuda:0")
        #     # coords_padding = torch.cat([coords,padding_coords],dim=0)
        
        # features_raw = features_padding.detach().cpu().numpy().astype(np.float32)
        # features_raw.tofile(os.path.join("/mnt/hdd/snpe_228/pointpillar_6490_a9_v2/pillar_train_pfe_ground_points", "pillar_" + batch_dict["frame_id"][0] +".raw"))

        for pfn in self.pfn_layers:
            features = pfn(features)
            
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict

class PillarVFE_QAT(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer_QAT(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0] # 0.16
        self.voxel_y = voxel_size[1] # 0.16
        self.voxel_z = voxel_size[2] # 4
        
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #  -> 0.08 + 0
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1] #  -> 0.08 + -80 
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2] #  -> -1
        
        
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0): # max_num : 32
        
        """
        Create boolean mask by actually number of a padded tensor.

            Args:
                actual_num ([type]): [description] -> actual number of points per voxel
                max_num ([type]): [description] -> 

            Returns:
                [type]: [description]  
        """    
        
        actual_num = torch.unsqueeze(actual_num, axis + 1) 
        # tiled_actual_num: [N, M, 1]
        
        max_num_shape = [1] * len(actual_num.shape) # [1, 1]
       
        max_num_shape[axis + 1] = -1 # [1, -1]

        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        
        
        # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
        # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
        
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]

        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # print(voxel_num_points.shape)
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean 
        
        f_center = torch.zeros_like(voxel_features[:, :, :3]) 
        
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
            
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
            
        features = torch.cat(features, dim=-1)
        
        voxel_count = features.shape[1]
        
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        
        features *= mask 
        if features.shape[0] <= 10000:
                padding = torch.zeros([10000 - features.shape[0],features.shape[1],features.shape[2]],dtype = torch.float32).to("cuda:0")
                features_padding = torch.cat([features,padding],dim = 0)

                # padding_coords = torch.zeros([10000-coords.shape[0], 4],dtype = torch.int32).to("cuda:0")
                # coords_padding = torch.cat([coords,padding_coords],dim=0)

        # features_raw = features_padding.detach().cpu().numpy().astype(np.float32)
        # features_raw.tofile(os.path.join("/media/ivpg/8TB/PJW/snpe_228/pointpillar_6490_a9_v2/pillar_train_histogram_QAT_without_norm", "pillar_" + batch_dict["frame_id"][0] +".raw"))
        
        for pfn in self.pfn_layers:
            features = pfn(features)
            
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict
    
class PillarVFE_QAT_input_norm(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer_QAT(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0] # 0.16
        self.voxel_y = voxel_size[1] # 0.16
        self.voxel_z = voxel_size[2] # 4
        
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #  -> 0.08 + 0
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1] #  -> 0.08 + -80 
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2] #  -> -1
        
        
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0): # max_num : 32
        
        """
        Create boolean mask by actually number of a padded tensor.

            Args:
                actual_num ([type]): [description] -> actual number of points per voxel
                max_num ([type]): [description] -> 

            Returns:
                [type]: [description]  
        """    
        
        actual_num = torch.unsqueeze(actual_num, axis + 1) 
        # tiled_actual_num: [N, M, 1]
        
        max_num_shape = [1] * len(actual_num.shape) # [1, 1]
       
        max_num_shape[axis + 1] = -1 # [1, -1]

        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        
        
        # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
        # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
        
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]

        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # print(voxel_num_points.shape)
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean 
        
        f_center = torch.zeros_like(voxel_features[:, :, :3]) 
        
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
            
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
            
        features = torch.cat(features, dim=-1)
        
        voxel_count = features.shape[1]
        
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        
        features *= mask 


        features_min = features.min(dim=-1,keepdim=True).values
        features_max = features.max(dim=-1,keepdim=True).values
        
        features_normalized = (features - features_min) / (features_max - features_min + 1e-8)

        if features.shape[0] <= 10000:
                padding = torch.zeros([10000 - features.shape[0],features.shape[1],features.shape[2]],dtype = torch.float32).to("cuda:0")
                features_padding = torch.cat([features,padding],dim = 0)

                # padding_coords = torch.zeros([10000-coords.shape[0], 4],dtype = torch.int32).to("cuda:0")
                # coords_padding = torch.cat([coords,padding_coords],dim=0)

        features_raw = features_padding.detach().cpu().numpy().astype(np.float32)
        features_raw.tofile(os.path.join("/media/ivpg/8TB/PJW/snpe_228/pointpillar_6490_a9_v2/pillar_train_histogram_QAT_input_norm", "pillar_" + batch_dict["frame_id"][0] +".raw"))
        
        for pfn in self.pfn_layers:
            features_normalized = pfn(features_normalized)
            
        features_normalized = features_normalized.squeeze()
        batch_dict['pillar_features'] = features_normalized
        return batch_dict


class PillarVFE_QAT_1d(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer_QAT_1d(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0] # 0.16
        self.voxel_y = voxel_size[1] # 0.16
        self.voxel_z = voxel_size[2] # 4
        
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #  -> 0.08 + 0
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1] #  -> 0.08 + -80 
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2] #  -> -1
        
        
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0): # max_num : 32
        
        """
        Create boolean mask by actually number of a padded tensor.

            Args:
                actual_num ([type]): [description] -> actual number of points per voxel
                max_num ([type]): [description] -> 

            Returns:
                [type]: [description]  
        """    
        
        actual_num = torch.unsqueeze(actual_num, axis + 1) 
        # tiled_actual_num: [N, M, 1]
        
        max_num_shape = [1] * len(actual_num.shape) # [1, 1]
       
        max_num_shape[axis + 1] = -1 # [1, -1]

        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        
        
        # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
        # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
        
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]

        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # print(voxel_num_points.shape)
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean 
        
        f_center = torch.zeros_like(voxel_features[:, :, :3]) 
        
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
            
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
            
        features = torch.cat(features, dim=-1)
        
        voxel_count = features.shape[1]
        
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        
        features *= mask 

        for pfn in self.pfn_layers:
            features = pfn(features)
            
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict
    
class PillarVFE_input_norm(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0] # 0.16
        self.voxel_y = voxel_size[1] # 0.16
        self.voxel_z = voxel_size[2] # 4
        
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #  -> 0.08 + 0
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1] #  -> 0.08 + -80 
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2] #  -> -1
        
        
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0): # max_num : 32
        
        """
        Create boolean mask by actually number of a padded tensor.

            Args:
                actual_num ([type]): [description] -> actual number of points per voxel
                max_num ([type]): [description] -> 

            Returns:
                [type]: [description]  
        """    
        
        actual_num = torch.unsqueeze(actual_num, axis + 1) 
        # tiled_actual_num: [N, M, 1]
        
        max_num_shape = [1] * len(actual_num.shape) # [1, 1]
       
        max_num_shape[axis + 1] = -1 # [1, -1]

        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        
        
        # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
        # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
        
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]

        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # print(voxel_num_points.shape)
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean 
        
        f_center = torch.zeros_like(voxel_features[:, :, :3]) 
        
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
            
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
            
        features = torch.cat(features, dim=-1)
        
        voxel_count = features.shape[1]
        
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        
        features *= mask 

        features_min = features.min(dim=-1,keepdim=True).values
        features_max = features.max(dim=-1,keepdim=True).values
        
        features_normalized = (features - features_min) / (features_max - features_min + 1e-8)
        
        for pfn in self.pfn_layers:
            features = pfn(features_normalized)
            
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict


class PillarVFE_input_norm_mean_plus(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0] # 0.16
        self.voxel_y = voxel_size[1] # 0.16
        self.voxel_z = voxel_size[2] # 4
        
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #  -> 0.08 + 0
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1] #  -> 0.08 + -80 
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2] #  -> -1
        
        
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0): # max_num : 32
        
        """
        Create boolean mask by actually number of a padded tensor.

            Args:
                actual_num ([type]): [description] -> actual number of points per voxel
                max_num ([type]): [description] -> 

            Returns:
                [type]: [description]  
        """    
        
        actual_num = torch.unsqueeze(actual_num, axis + 1) 
        # tiled_actual_num: [N, M, 1]
        
        max_num_shape = [1] * len(actual_num.shape) # [1, 1]
       
        max_num_shape[axis + 1] = -1 # [1, -1]

        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        
        
        # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
        # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
        
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]

        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # print(voxel_num_points.shape)
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean 
        
        f_center = torch.zeros_like(voxel_features[:, :, :3]) 
        
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
            
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
            
        features = torch.cat(features, dim=-1)
        
        voxel_count = features.shape[1]
        
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        
        features *= mask 

        features_min = features.min(dim=-1,keepdim=True).values
        features_max = features.max(dim=-1,keepdim=True).values
        
        features_normalized = (features - features_min) / (features_max - features_min + 1e-8)
        
        zero_channel_mask = torch.all(features_normalized == 0, dim=2, keepdim=True)  # shape: (16000, 32, 1)

        # 모든 채널 값이 0이 아닌 위치 확인 (non-zero locations)
        non_zero_channel_mask = ~zero_channel_mask  # 반전하여 non-zero 위치 확인

        # non-zero 위치에서 각 채널 평균 계산
        # 채널 축(2) 기준으로 값이 0이 아닌 위치의 평균 계산
        non_zero_sums = torch.sum(features_normalized * non_zero_channel_mask, dim=0)  # shape: (32, 10)
        non_zero_counts = torch.sum(non_zero_channel_mask, dim=0)  # shape: (32, 10)
        non_zero_means = non_zero_sums / torch.clamp(non_zero_counts, min=1)  # shape: (32, 10)

        # zero location에 채널별 평균값 대입
        features_normalized = torch.where(zero_channel_mask, non_zero_means.unsqueeze(0), features_normalized)

        for pfn in self.pfn_layers:
            features = pfn(features_normalized)
            
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        
        return batch_dict
    
class PillarVFE_mean_plus(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0] # 0.16
        self.voxel_y = voxel_size[1] # 0.16
        self.voxel_z = voxel_size[2] # 4
        
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #  -> 0.08 + 0
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1] #  -> 0.08 + -80 
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2] #  -> -1
        
        
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0): # max_num : 32
        
        """
        Create boolean mask by actually number of a padded tensor.

            Args:
                actual_num ([type]): [description] -> actual number of points per voxel
                max_num ([type]): [description] -> 

            Returns:
                [type]: [description]  
        """    
        
        actual_num = torch.unsqueeze(actual_num, axis + 1) 
        # tiled_actual_num: [N, M, 1]
        
        max_num_shape = [1] * len(actual_num.shape) # [1, 1]
       
        max_num_shape[axis + 1] = -1 # [1, -1]

        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        
        
        # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
        # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
        
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]

        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # print(voxel_num_points.shape)
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean 
        
        f_center = torch.zeros_like(voxel_features[:, :, :3]) 
        
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
            
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
            
        features = torch.cat(features, dim=-1)
        
        voxel_count = features.shape[1]
        
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        
        features *= mask 
        
        # if features.shape[0] <= 10000:
            
        #     padding = torch.zeros([10000 - features.shape[0],features.shape[1],features.shape[2]],dtype = torch.float32).to("cuda:0")
        #     features = torch.cat([features,padding],dim = 0)
            
        #     padding_num_points = torch.zeros([10000 - voxel_num_points.shape[0]],dtype = torch.float32) .to(voxel_num_points.device)
        #     voxel_num_points = torch.cat([voxel_num_points,padding_num_points],dim = 0)
        #     batch_dict['voxel_num_points'] = voxel_num_points

        #     padding_coords = torch.zeros([10000 - coords.shape[0],coords.shape[1]],dtype = torch.float32).to(coords.device)
        #     coords = torch.cat([coords,padding_coords],dim = 0)
        #     batch_dict['voxel_coords'] = coords
            
        # features_raw = features.detach().cpu().numpy().astype(np.float32)
        # features_raw.tofile(os.path.join("/mnt/hdd/snpe_228/pointpillar_6490_a9_v2/pillar_train_pfe_mean_plus", "pillar_" + batch_dict["frame_id"][0] +".raw"))
        
        zero_channel_mask = torch.all(features == 0, dim=2, keepdim=True)  # shape: (16000, 32, 1)

        # 모든 채널 값이 0이 아닌 위치 확인 (non-zero locations)
        non_zero_channel_mask = ~zero_channel_mask  # 반전하여 non-zero 위치 확인

        # non-zero 위치에서 각 채널 평균 계산
        # 채널 축(2) 기준으로 값이 0이 아닌 위치의 평균 계산
        non_zero_sums = torch.sum(features * non_zero_channel_mask, dim=0)  # shape: (32, 10)
        non_zero_counts = torch.sum(non_zero_channel_mask, dim=0)  # shape: (32, 10)
        non_zero_means = non_zero_sums / torch.clamp(non_zero_counts, min=1)  # shape: (32, 10)

        # zero location에 채널별 평균값 대입
        features = torch.where(zero_channel_mask, non_zero_means.unsqueeze(0), features)

        for pfn in self.pfn_layers:
            features = pfn(features)
            
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        
        return batch_dict
    
class PillarVFE_1x1conv(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer_1x1conv(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0] # 0.16
        self.voxel_y = voxel_size[1] # 0.16
        self.voxel_z = voxel_size[2] # 4
        
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #  -> 0.08 + 0
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1] #  -> 0.08 + -80 
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2] #  -> -1
        
        
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0): # max_num : 32
        
        """
        Create boolean mask by actually number of a padded tensor.

            Args:
                actual_num ([type]): [description] -> actual number of points per voxel
                max_num ([type]): [description] -> 

            Returns:
                [type]: [description]  
        """    
        
        actual_num = torch.unsqueeze(actual_num, axis + 1) 
        # tiled_actual_num: [N, M, 1]
        
        max_num_shape = [1] * len(actual_num.shape) # [1, 1]
       
        max_num_shape[axis + 1] = -1 # [1, -1]

        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        
        
        # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
        # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
        
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]

        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # print(voxel_num_points.shape)
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean 
        
        f_center = torch.zeros_like(voxel_features[:, :, :3]) 
        
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
            
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
            
        features = torch.cat(features, dim=-1)
        
        voxel_count = features.shape[1]
        
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        
        features *= mask 
        
        # features_min = features.min(dim=-1,keepdim=True).values
        # features_max = features.max(dim=-1,keepdim=True).values
        
        # features_normalized = (features - features_min) / (features_max - features_min + 1e-8)
        
        # if features.shape[0] <= 10000:

        #     padding = torch.zeros([10000 - features.shape[0],features.shape[1],features.shape[2]],dtype = torch.float32).to("cuda:0")
        #     features_padding = torch.cat([features,padding],dim = 0)

        #     padding_coords = torch.zeros([10000-coords.shape[0], 4],dtype = torch.int32).to("cuda:0")
        #     coords_padding = torch.cat([coords,padding_coords],dim=0)
        
        # print(coords_padding.shape)
        # features_raw = features_padding.detach().cpu().numpy().astype(np.float32)
        # features_raw.tofile(os.path.join("/media/ivpg/8TB/PJW/snpe_228/pointpillar_6490_a9_v2/pillar_test_1x1conv", "pillar_" + batch_dict["frame_id"][0] +".raw"))

        # voxel_coords_raw = coords_padding.detach().cpu().numpy().astype(np.float32)
        # voxel_coords_raw.tofile(os.path.join("/media/ivpg/8TB/PJW/snpe_228/pointpillar_6490_a9_v2/coord_test_1x1conv", "voxel_coords_" + batch_dict["frame_id"][0] +".raw"))
               
        
        for pfn in self.pfn_layers:
            features = pfn(features)
            
        features = features.squeeze()
        batch_dict['pillar_features'] = features

        return batch_dict

class PillarVFE_input_norm(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0] # 0.16
        self.voxel_y = voxel_size[1] # 0.16
        self.voxel_z = voxel_size[2] # 4
        
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #  -> 0.08 + 0
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1] #  -> 0.08 + -80 
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2] #  -> -1
        
        
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0): # max_num : 32
        
        """
        Create boolean mask by actually number of a padded tensor.

            Args:
                actual_num ([type]): [description] -> actual number of points per voxel
                max_num ([type]): [description] -> 

            Returns:
                [type]: [description]  
        """    
        
        actual_num = torch.unsqueeze(actual_num, axis + 1) 
        # tiled_actual_num: [N, M, 1]
        
        max_num_shape = [1] * len(actual_num.shape) # [1, 1]
       
        max_num_shape[axis + 1] = -1 # [1, -1]

        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        
        
        # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
        # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
        
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]

        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # print(voxel_num_points.shape)
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean 
        
        f_center = torch.zeros_like(voxel_features[:, :, :3]) 
        
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
            
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
            
        features = torch.cat(features, dim=-1)
        
        voxel_count = features.shape[1]
        
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        
        features *= mask 

        features_min = features.min(dim=-1,keepdim=True).values
        features_max = features.max(dim=-1,keepdim=True).values
        
        features_normalized = (features - features_min) / (features_max - features_min + 1e-8)
        
        # if features_normalized.shape[0] <= 10000:
        #       padding = torch.zeros([10000 - features_normalized.shape[0],features_normalized.shape[1],features_normalized.shape[2]],dtype = torch.float32).to("cuda:0")
        #       features_padding = torch.cat([features_normalized,padding],dim = 0)

                # padding_coords = torch.zeros([10000-coords.shape[0], 4],dtype = torch.int32).to("cuda:0")
                # coords_padding = torch.cat([coords,padding_coords],dim=0)
            
        # features_raw = features_padding.detach().cpu().numpy().astype(np.float32)
        # features_raw.tofile(os.path.join("/media/ivpg/8TB/PJW/snpe_228/pointpillar_6490_a9_v2/pillar_train_histogram_QAT_input_norm", "pillar_" + batch_dict["frame_id"][0] +".raw"))
                
        for pfn in self.pfn_layers:
            features = pfn(features_normalized)
            
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict
    
class PillarVFE_complex_yolo(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0] # 0.16
        self.voxel_y = voxel_size[1] # 0.16
        self.voxel_z = voxel_size[2] # 4
        
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #  -> 0.08 + 0
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1] #  -> 0.08 + -80 
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2] #  -> -1
        
        
    def get_output_feature_dim(self):
        return self.num_filters[-1]
    
    def get_paddings_indicator(self, actual_num, max_num, axis=0): # max_num : 32
        
        """
        Create boolean mask by actually number of a padded tensor.

            Args:
                actual_num ([type]): [description] -> actual number of points per voxel
                max_num ([type]): [description] -> 

            Returns:
                [type]: [description]  
        """    
        
        actual_num = torch.unsqueeze(actual_num, axis + 1) 
        # tiled_actual_num: [N, M, 1]
        
        max_num_shape = [1] * len(actual_num.shape) # [1, 1]
       
        max_num_shape[axis + 1] = -1 # [1, -1]

        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        
        
        # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
        # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
        
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]

        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        
        # print(torch.min(batch_dict['points'][:, 2]),torch.max(batch_dict['points'][:, 2]))

        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']

        # print('coords', torch.max(coords[:, 3]))
        # print(voxel_num_points.shape)
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean 
        
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        
        f_center[:, :, 0] = coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset
        f_center[:, :, 1] = coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset
        f_center[:, :, 2] = coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset
        print(f_center[:, :, 2])
        mean_z_per_pillar = torch.mean(f_center[:, :, 2], dim=1)

        # if self.use_absolute_xyz:
        #     features = [voxel_features, f_cluster, f_center]
        # else:
        #     features = [voxel_features[..., 3:], f_cluster, f_center]
            
        # if self.with_distance:
        #     points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
        #     features.append(points_dist)
            
        # features = torch.cat(features, dim=-1)
        
        # voxel_count = features.shape[1]
        
        # mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        # mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        
        # features *= mask
        # f_center *= mask 
            
        # for pfn in self.pfn_layers:
        #     features = pfn(features)
            
        # features = features.squeeze()
        # batch_dict['pillar_features'] = features[:, :, 7:9]
        batch_dict['pillar_features'] = mean_z_per_pillar
        return batch_dict

class PillarVFE_skip(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        # self.use_norm = self.model_cfg.USE_NORM
        # self.with_distance = self.model_cfg.WITH_DISTANCE
        # self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        # num_point_features += 6 if self.use_absolute_xyz else 3
        # if self.with_distance:
        #     num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        # num_filters = [num_point_features] + list(self.num_filters)

        # pfn_layers = []
        # for i in range(len(num_filters) - 1):
        #     in_filters = num_filters[i]
        #     out_filters = num_filters[i + 1]
        #     pfn_layers.append(
        #         PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
        #     )
        # self.pfn_layers = nn.ModuleList(pfn_layers)

        # self.voxel_x = voxel_size[0] # 0.16
        # self.voxel_y = voxel_size[1] # 0.16
        # self.voxel_z = voxel_size[2] # 4
        
        # self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #  -> 0.08 + 0
        # self.y_offset = self.voxel_y / 2 + point_cloud_range[1] #  -> 0.08 + -80 
        # self.z_offset = self.voxel_z / 2 + point_cloud_range[2] #  -> -1
        
        
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    # def get_paddings_indicator(self, actual_num, max_num, axis=0): # max_num : 32
        
    #     """
    #     Create boolean mask by actually number of a padded tensor.

    #         Args:
    #             actual_num ([type]): [description] -> actual number of points per voxel
    #             max_num ([type]): [description] -> 

    #         Returns:
    #             [type]: [description]  
    #     """    
        
    #     actual_num = torch.unsqueeze(actual_num, axis + 1) 
    #     # tiled_actual_num: [N, M, 1]
        
    #     max_num_shape = [1] * len(actual_num.shape) # [1, 1]
       
    #     max_num_shape[axis + 1] = -1 # [1, -1]

    #     max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        
        
    #     # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    #     # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
        
    #     paddings_indicator = actual_num.int() > max_num
    #     # paddings_indicator shape: [batch_size, max_num]

    #     return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        # print(batch_dict.keys())
        return batch_dict
    
class Simple3DPointNet(nn.Module):
    def __init__(self,input_dim,num_filters):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels=input_dim, out_channels=num_filters, kernel_size=1, stride=1, padding=0,bias=False)
        self.bn1 = nn.BatchNorm3d(num_filters,eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU()
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = torch.max(x, dim=2, keepdim=True)[0].squeeze(dim = 2)
        
        return x
    
class Pillar_to_conv3D(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        self.nx, self.ny, self.nz = self.model_cfg.GRID_SIZE
        self.input_dim = self.model_cfg.INPUT_DIM
        
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        
        self.pointnet_conv_3d_layers = Simple3DPointNet(input_dim = self.input_dim,num_filters= self.num_filters[0])
        
        self.voxel_x = voxel_size[0] # 0.16
        self.voxel_y = voxel_size[1] # 0.16
        self.voxel_z = voxel_size[2] # 4
        
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #  -> 0.08 + 0
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1] #  -> 0.08 + -80 
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2] #  -> -1
        
        
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0): # max_num : 32
        
        """
        Create boolean mask by actually number of a padded tensor.

            Args:
                actual_num ([type]): [description] -> actual number of points per voxel
                max_num ([type]): [description] -> 

            Returns:
                [type]: [description]  
        """    
        
        actual_num = torch.unsqueeze(actual_num, axis + 1) 
        # tiled_actual_num: [N, M, 1]
        
        max_num_shape = [1] * len(actual_num.shape) # [1, 1]
       
        max_num_shape[axis + 1] = -1 # [1, -1]

        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        
        # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
        # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
        
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]

        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        
        f_cluster = voxel_features[:, :, :3] - points_mean 
        
        f_center = torch.zeros_like(voxel_features[:, :, :3]) 
        
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
            
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
            
        features = torch.cat(features, dim=-1)
        
        voxel_count = features.shape[1]
        
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        
        features *= mask 
        batch_spatial_features = []
        max_num_voxel = features.shape[1]
        
        batch_size = coords[:, 0].max().int().item() + 1
        
        for batch_idx in range(batch_size):
            
            spatial_feature = torch.zeros(
                self.input_dim,
                max_num_voxel,
                self.nz * self.nx * self.ny,
                dtype=features.dtype,
                device=features.device)
            
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = features[batch_mask, :]
            pillars = pillars.permute(2,1,0)
            spatial_feature[:, :, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.input_dim, max_num_voxel * self.nz, self.ny, self.nx)
        
        # print(batch_spatial_features.shape) # 2 x 10 x 32 x 608 x 608
        batch_pseudo_image = self.pointnet_conv_3d_layers(batch_spatial_features)
        batch_dict['spatial_features'] = batch_pseudo_image

        return batch_dict


class PillarVFE_onnx(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)


    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        
        max_num_shape = [1] * len(actual_num.shape)
        
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num

        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
  
    
        features = batch_dict['pillar_features']
        # features = features.squeeze()
        
        for pfn in self.pfn_layers:
            features = pfn(features)
            
        # features = features.squeeze().permute(1,0) # for 1 x 1 conv
        
        features = features.squeeze()
        
        # print(features.shape)

        batch_dict['pillar_features'] = features
        return batch_dict

class PillarVFE_dual(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        
        pfn_layers_background = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers_background.append(
                PFNLayer_backgrond(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
            
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.pfn_layers_background = nn.ModuleList(pfn_layers_background)
        
        self.voxel_x = voxel_size[0] # 0.16
        self.voxel_y = voxel_size[1] # 0.16
        self.voxel_z = voxel_size[2] # 4
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #  -> 0.08
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1] #  -> -39.6
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2] #  -> -1
        
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0): # max_num : 32
        
        """
        Create boolean mask by actually number of a padded tensor.

            Args:
                actual_num ([type]): [description] -> actual number of points per voxel
                max_num ([type]): [description] -> 

            Returns:
                [type]: [description]  
        """    
        
        actual_num = torch.unsqueeze(actual_num, axis + 1) 
        # tiled_actual_num: [N, M, 1]
        
        max_num_shape = [1] * len(actual_num.shape) # [1, 1]
       
        max_num_shape[axis + 1] = -1 # [1, -1]

        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        
        
        # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
        # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
        
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]

        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']

        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean 
        
        f_center = torch.zeros_like(voxel_features[:, :, :3]) 
        
        # print(f_center.shape)
        # print(voxel_features.shape)
        # print(coords.shape)
        # print(voxel_num_points.shape)
        
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
            
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
            
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        
        features *= mask 
        
        #### background generating pillar ####
        
        voxel_features, voxel_num_points, coords = batch_dict['voxels_background'], batch_dict['voxel_num_points_background'], batch_dict['voxel_coords_background']

        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean 
        
        f_center = torch.zeros_like(voxel_features[:, :, :3]) 
        
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            background_features = [voxel_features, f_cluster, f_center]
        else:
            background_features = [voxel_features[..., 3:], f_cluster, f_center]
            
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            background_features.append(points_dist)
            
        background_features = torch.cat(background_features, dim=-1)

        voxel_count = background_features.shape[1]
        
        background_mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        background_mask = torch.unsqueeze(background_mask, -1).type_as(voxel_features)
        
        background_features *= background_mask 
        
        #### background generating pillar ####
        
        for pfn in self.pfn_layers:
            features = pfn(features)
            
        for pfn in self.pfn_layers_background:
            background_features = pfn(background_features)  
            
        features = features.squeeze()
        background_features = background_features.squeeze()
        
        batch_dict['pillar_features'] = features
        batch_dict['pillar_background_features'] = background_features
        
        return batch_dict
    
class PillarVFE_forground_extraction(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )

        self.point_cloud_range = point_cloud_range
        self.pfn_layers = nn.ModuleList(pfn_layers)
        
        self.voxel_x = voxel_size[0] # 0.16
        self.voxel_y = voxel_size[1] # 0.16
        self.voxel_z = voxel_size[2] # 4
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #  -> 0.08
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1] #  -> -39.6
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2] #  -> -1

        self.octree_voxel_size = 0.01 
        
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0): # max_num : 32
        
        """
        Create boolean mask by actually number of a padded tensor.

            Args:
                actual_num ([type]): [description] -> actual number of points per voxel
                max_num ([type]): [description] -> 

            Returns:
                [type]: [description]  
        """    
        
        actual_num = torch.unsqueeze(actual_num, axis + 1) 
        # tiled_actual_num: [N, M, 1]
        
        max_num_shape = [1] * len(actual_num.shape) # [1, 1]
       
        max_num_shape[axis + 1] = -1 # [1, -1]

        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        
        
        # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
        # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
        
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape: [batch_size, max_num]

        return paddings_indicator
    
    def extract_foreground_voxelgrid_batch(self, background_pointcloud, current_pointcloud, voxel_size):
        """
        수정사항: 전경으로 추출된 포인트 중 Z > -5 (공중 노이즈)인 점들을 제거하고,
                 Z <= -5 (지면 근처)인 점들만 남김.
        """
        
        # 텐서가 GPU에 있다면 .cpu()로 이동
        if background_pointcloud.is_cuda:
            background_pointcloud = background_pointcloud.cpu()
        if current_pointcloud.is_cuda:
            current_pointcloud = current_pointcloud.cpu()

        batch_indices = torch.unique(current_pointcloud[:, 0]).int()
        foreground_pcd_list = []

        for b_idx in batch_indices:
            # 1. 현재 배치 인덱스(b_idx)에 해당하는 포인트들만 필터링
            bg_mask = (background_pointcloud[:, 0] == b_idx)
            curr_mask = (current_pointcloud[:, 0] == b_idx)
            
            bg_points_np = background_pointcloud[bg_mask, 1:4].numpy()
            
            # 현재 씬 데이터 준비
            curr_batch_data = current_pointcloud[curr_mask]
            curr_points_np = curr_batch_data[:, 1:4].numpy()     # X, Y, Z
            curr_batch_data_np = curr_batch_data.numpy()         # B, X, Y, Z, I

            if curr_points_np.shape[0] == 0:
                foreground_pcd_list.append(o3d.geometry.PointCloud())
                continue

            pcd_bg = o3d.geometry.PointCloud()
            pcd_bg.points = o3d.utility.Vector3dVector(bg_points_np)
            
            # =========================================================
            # Case 1: 배경이 비어있는 경우 (모든 점이 전경 후보)
            # =========================================================
            if bg_points_np.shape[0] == 0:

                valid_z_mask = curr_batch_data_np[:, 3] <= -3.0
                
                filtered_points = curr_points_np[valid_z_mask]      # XYZ
                filtered_data = curr_batch_data_np[valid_z_mask]    # Full Data

                pcd_fg = o3d.geometry.PointCloud()
                pcd_fg.points = o3d.utility.Vector3dVector(filtered_points)
                
                if filtered_data.shape[0] > 0:
                    intensity = filtered_data[:, 4:5]
                    pcd_fg.colors = o3d.utility.Vector3dVector(np.stack([intensity[:,0]]*3, axis=-1))
                
                foreground_pcd_list.append(pcd_fg)
                continue

            # =========================================================
            # Case 2: 배경이 있는 경우 (복셀 비교 수행)
            # =========================================================
            voxel_bg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_bg, voxel_size)
            bg_voxel_indices = {tuple(v.grid_index) for v in voxel_bg.get_voxels()}
            origin = voxel_bg.origin

            foreground_point_indices = []
            for i in range(curr_points_np.shape[0]):
                point = curr_points_np[i]
                
                # 1차: 배경 복셀에 속하는지 확인
                grid_index = tuple(np.floor((point - origin) / voxel_size).astype(int))
                
                if grid_index not in bg_voxel_indices:
                    # 배경에 없으면 일단 전경 후보로 추가
                    foreground_point_indices.append(i)
            
            # 전경 후보들에 대해 후처리 (Z축 필터링) 수행
            if foreground_point_indices:
                fg_points_data = curr_batch_data_np[foreground_point_indices] # 전경 후보들의 전체 데이터
                
                # [핵심 수정] 전경으로 판별된 점들 중, Z > -5 (공중)인 점 제거
                # 즉, Z <= -5 인 점만 남김
                valid_z_mask = fg_points_data[:, 3] <= -5.0
                fg_points_data = fg_points_data[valid_z_mask]
                
                # 필터링 후 점이 남아있는지 확인
                if fg_points_data.shape[0] > 0:
                    pcd_fg = o3d.geometry.PointCloud()
                    pcd_fg.points = o3d.utility.Vector3dVector(fg_points_data[:, 1:4]) # X, Y, Z
                    
                    # Intensity 시각화
                    intensity_colors = np.clip(fg_points_data[:, 4], 0, 1)
                    pcd_fg.colors = o3d.utility.Vector3dVector(
                        np.stack([intensity_colors]*3, axis=-1)
                    )
                    foreground_pcd_list.append(pcd_fg)
                else:
                    # 전경 후보가 있었으나 Z 필터링으로 다 사라진 경우
                    foreground_pcd_list.append(o3d.geometry.PointCloud())
            else:
                # 배경 차분 결과 전경이 없는 경우
                foreground_pcd_list.append(o3d.geometry.PointCloud())

        return foreground_pcd_list
    
    def create_bev_mask_batch(self, foreground_pcd_list: List[o3d.geometry.PointCloud], 
                                batch_size: int, device: torch.device):
            """
            배치별 전경 포인트 클라우드 리스트를 (B, 1, H, W) 2D BEV 마스크로 스캐터링합니다.

            Args:
                foreground_pcd_list (list[o3d.geometry.PointCloud]): 
                    배치별 전경 PointCloud 리스트
                batch_size (int): 
                    전체 배치 크기 (B)
                device (torch.device): 
                    생성될 텐서가 위치할 디바이스 (e.g., 'cuda:0')

            Returns:
                torch.Tensor: 
                    Shape (B, 1, H, W)의 바이너리 마스크 텐서
            """
            
            # 그리드 파라미터 (H, W 계산)
            pc_range = self.point_cloud_range
            min_x, min_y = pc_range[0], pc_range[1]
            size_x, size_y = self.voxel_x, self.voxel_y
            
            # H (grid_size_x)
            grid_size_x = int(np.round((pc_range[3] - pc_range[0]) / size_x)) 
            # W (grid_size_y)
            grid_size_y = int(np.round((pc_range[4] - pc_range[1]) / size_y)) 
            
            # (B, 1, H, W) 크기의 0으로 채워진 텐서 초기화
            output_mask_tensor = torch.zeros((batch_size, 1, grid_size_x, grid_size_y), 
                                            dtype=torch.float32, device=device)

            # 리스트의 순서(i)가 곧 배치의 순서
            for i in range(batch_size):
                fg_points_np = np.asarray(foreground_pcd_list[i].points)
                
                if fg_points_np.shape[0] == 0:
                    continue # 현재 배치는 전경이 없음
                    
                # 1. 전경 포인트 좌표를 (H, W) 그리드 인덱스로 변환
                x_indices = np.floor((fg_points_np[:, 0] - min_x) / size_x).astype(int)
                y_indices = np.floor((fg_points_np[:, 1] - min_y) / size_y).astype(int)
                
                # 2. 그리드 범위 밖의 포인트들 필터링
                valid_mask = (x_indices >= 0) & (x_indices < grid_size_x) & \
                            (y_indices >= 0) & (y_indices < grid_size_y)
                
                if not np.any(valid_mask):
                    continue

                x_indices_valid = x_indices[valid_mask]
                y_indices_valid = y_indices[valid_mask]
                
                # 3. 인덱스를 텐서로 변환
                x_indices_torch = torch.from_numpy(x_indices_valid).long().to(device)
                y_indices_torch = torch.from_numpy(y_indices_valid).long().to(device)
                
                # 4. (B, 1, H, W) 텐서의 (i, 0, H_idx, W_idx) 위치에 '1'을 스캐터링
                output_mask_tensor[i, 0, x_indices_torch, y_indices_torch] = 1.0
                
            return output_mask_tensor
    
    def forward(self, batch_dict, **kwargs):
        
        # print(batch_dict['points'])
        # print(batch_dict['background'])
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']

        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean 
        
        f_center = torch.zeros_like(voxel_features[:, :, :3]) 
        
        # print(f_center.shape)
        # print(voxel_features.shape)
        # print(coords.shape)
        # print(voxel_num_points.shape)
        
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
            
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
            
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        
        features *= mask 
        
        #### foreground extraction pillar ####

        background_pointcloud = batch_dict['background'] # B,X,Y,Z,i
        current_pointcloud = batch_dict['points'] # B,X,Y,Z,i

        foreground_results = self.extract_foreground_voxelgrid_batch(
            background_pointcloud, 
            current_pointcloud, 
            voxel_size=self.octree_voxel_size
        )

        foreground_bev_mask = self.create_bev_mask_batch(
            foreground_results,
            batch_dict['batch_size'],
            batch_dict['background'].device
        )

        ###### foreground Visualization #######
        # print(batch_dict["frame_id"])
        # mask = (background_pointcloud[:, 0] == 0)
        # batch_0_points = background_pointcloud[mask] # B, x, y, z, i


        # mask = (background_pointcloud[:, 0] == 1)
        # batch_1_points = background_pointcloud[mask]

        # # --- 결과 확인 ---
        # print(f"총 {len(foreground_results)}개의 배치가 처리되었습니다.")

        # # 배치 0의 결과 (박스만 나와야 함)
        # print(f"\n배치 0 (전경) 포인트 개수: {len(foreground_results[0].points)}")
        # if len(foreground_results[0].points) > 0:
        #     # (시각화) 첫 번째 배치의 전경(박스)을 빨간색으로, 배경을 회색으로 표시
        #     pcd_b0_bg = o3d.geometry.PointCloud()
        #     pcd_b0_bg.points = o3d.utility.Vector3dVector(batch_0_points.cpu().numpy()[:,1:4])
        #     pcd_b0_bg.paint_uniform_color([0.5, 0.5, 0.5]) # 회색
            
        #     # 전경은 반사도(회색조) 대신 빨간색으로 칠해서 확인
        #     foreground_results[0].paint_uniform_color([1.0, 0, 0]) # 빨간색
            
        #     o3d.visualization.draw_geometries(
        #         [foreground_results[0], pcd_b0_bg], 
        #         window_name="Batch 0 - Foreground (Red) vs Background (Gray)"
        #     )

        # # 배치 1의 결과 (포인트가 없어야 함)
        # print(f"\n배치 1 (전경) 포인트 개수: {len(foreground_results[1].points)}")
        # if len(foreground_results[1].points) > 0:
        #     # (시각화) 첫 번째 배치의 전경(박스)을 빨간색으로, 배경을 회색으로 표시
        #     pcd_b1_bg = o3d.geometry.PointCloud()
        #     pcd_b1_bg.points = o3d.utility.Vector3dVector(batch_1_points.cpu().numpy()[:,1:4])
        #     pcd_b1_bg.paint_uniform_color([0.5, 0.5, 0.5]) # 회색
            
        #     # 전경은 반사도(회색조) 대신 빨간색으로 칠해서 확인
        #     foreground_results[1].paint_uniform_color([1.0, 0, 0]) # 빨간색
            
        #     o3d.visualization.draw_geometries(
        #         [foreground_results[1], pcd_b1_bg], 
        #         window_name="Batch 1 - Foreground (Red) vs Background (Gray)"
        #     )

        #### background generating pillar ####

        for pfn in self.pfn_layers:
            features = pfn(features)
            
        features = features.squeeze()
        
        batch_dict['pillar_features'] = features
        batch_dict['foreground_bev_mask'] = foreground_bev_mask
        
        return batch_dict
    


class PillarVFE_forground_extraction_dual_pointnet(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        
        # ==========================================
        # 1. Main Branch 설정 (기존 PillarVFE)
        # ==========================================
        num_point_features_main = num_point_features
        num_point_features_main += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features_main += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features_main] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # ==========================================
        # 2. Foreground Branch 설정 (Single MLP PointNet)
        # ==========================================
        # PointPillars의 정석 입력 차원은 10D 입니다.
        # (x, y, z, i, x_c, y_c, z_c, x_p, y_p, d)
        # - x, y, z, i: 원본 포인트
        # - x_c, y_c, z_c: Cluster(Pillar) 내 평균과의 거리 (점 1개이므로 0)
        # - x_p, y_p, z_p: Pillar Center(Voxel Center)와의 거리
        # - d: 원점과의 거리 (Optional)
        
        self.fg_input_dim = 10  # 정석대로 10차원으로 고정
        self.fg_feature_dim = 64 # 출력은 64차원

        # [핵심] 단 하나의 PFNLayer 사용 (10 -> 64)
        # last_layer=True로 설정하여 Linear -> BN -> ReLU -> MaxPool 실행
        self.fg_pfn = PFNLayer(
            in_channels=self.fg_input_dim, 
            out_channels=self.fg_feature_dim, 
            use_norm=self.use_norm, 
            last_layer=True 
        )

        # ==========================================
        # 3. Grid Parameters
        # ==========================================
        self.point_cloud_range = point_cloud_range
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        
        self.octree_voxel_size = 0.01
        
        self.nx = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_x)
        self.ny = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_y)

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def extract_foreground_voxelgrid_batch(self, background_pointcloud, current_pointcloud, voxel_size):
        
        if background_pointcloud.is_cuda:
            background_pointcloud = background_pointcloud.cpu()
        if current_pointcloud.is_cuda:
            current_pointcloud = current_pointcloud.cpu()

        batch_indices = torch.unique(current_pointcloud[:, 0]).int()
        foreground_pcd_list = []

        for b_idx in batch_indices:
            bg_mask = (background_pointcloud[:, 0] == b_idx)
            curr_mask = (current_pointcloud[:, 0] == b_idx)
            
            bg_points_np = background_pointcloud[bg_mask, 1:4].numpy()
            curr_batch_data = current_pointcloud[curr_mask]
            curr_points_np = curr_batch_data[:, 1:4].numpy()
            curr_batch_data_np = curr_batch_data.numpy() # B, X, Y, Z, I

            if curr_points_np.shape[0] == 0:
                foreground_pcd_list.append(o3d.geometry.PointCloud())
                continue

            pcd_bg = o3d.geometry.PointCloud()
            pcd_bg.points = o3d.utility.Vector3dVector(bg_points_np)
            
            # -------------------------------------------------------
            # Case 1: 배경 데이터가 아예 없는 경우
            # -------------------------------------------------------
            if bg_points_np.shape[0] == 0:
                # [중요] "Z > -5"인 공중 노이즈를 제거 (즉, Z <= -5 만 남김)
                valid_z_mask = curr_points_np[:, 2] <= -3.0 
                
                filtered_points = curr_points_np[valid_z_mask]
                filtered_data = curr_batch_data_np[valid_z_mask]

                pcd_fg = o3d.geometry.PointCloud()
                pcd_fg.points = o3d.utility.Vector3dVector(filtered_points)
                
                if filtered_data.shape[0] > 0:
                    intensity = filtered_data[:, 4:5]
                    pcd_fg.colors = o3d.utility.Vector3dVector(np.tile(intensity, (1, 3)))
                foreground_pcd_list.append(pcd_fg)
                continue

            # -------------------------------------------------------
            # Case 2: 배경 데이터가 있어서 Voxel 비교를 하는 경우
            # -------------------------------------------------------
            voxel_bg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_bg, voxel_size)
            bg_voxel_indices = {tuple(v.grid_index) for v in voxel_bg.get_voxels()}
            origin = voxel_bg.origin

            foreground_point_indices = []
            for i in range(curr_points_np.shape[0]):
                point = curr_points_np[i]
                
                # 여기서는 일단 배경 복셀 여부만 판단 (좌표 계산)
                grid_index = tuple(np.floor((point - origin) / voxel_size).astype(int))
                
                if grid_index not in bg_voxel_indices:
                    foreground_point_indices.append(i)
            
            # 전경 후보가 추출된 후, Z축 필터링 적용
            if foreground_point_indices:
                fg_points_data = curr_batch_data_np[foreground_point_indices]
                
                # [중요] 여기서 Z > -5 인 점들을 제거 (Z <= -5 유지)
                # fg_points_data[:, 3] 은 Z좌표입니다.
                valid_z_mask = fg_points_data[:, 3] <= -5.0
                fg_points_data = fg_points_data[valid_z_mask]

                # 필터링 후 점이 남아있으면 PointCloud 생성
                if fg_points_data.shape[0] > 0:
                    pcd_fg = o3d.geometry.PointCloud()
                    pcd_fg.points = o3d.utility.Vector3dVector(fg_points_data[:, 1:4])
                    # Intensity 저장
                    intensity = fg_points_data[:, 4:5]
                    pcd_fg.colors = o3d.utility.Vector3dVector(np.tile(np.clip(intensity, 0, 1), (1, 3)))
                    foreground_pcd_list.append(pcd_fg)
                else:
                    foreground_pcd_list.append(o3d.geometry.PointCloud())
            else:
                foreground_pcd_list.append(o3d.geometry.PointCloud())

        return foreground_pcd_list

    def forward(self, batch_dict, **kwargs):
        
        # -----------------------------------------------------
        # 1. Main Branch (기존 PointPillars Logic)
        # -----------------------------------------------------
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean 
        
        f_center = torch.zeros_like(voxel_features[:, :, :3]) 
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
            
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
            
        features = torch.cat(features, dim=-1)
        
        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask 
        
        for pfn in self.pfn_layers:
            features = pfn(features)
        
        main_features = features.squeeze()
        batch_dict['pillar_features'] = main_features


        # -----------------------------------------------------
        # 2. Foreground Branch (Single MLP PointNet)
        # -----------------------------------------------------
        background_pointcloud = batch_dict['background']
        current_pointcloud = batch_dict['points']
        device = background_pointcloud.device

        # (1) Foreground Point Extraction (Open3D)
        foreground_pcd_list = self.extract_foreground_voxelgrid_batch(
            background_pointcloud, 
            current_pointcloud, 
            voxel_size=self.octree_voxel_size
        )

        # (2) Pillarization per Point & Single MLP & Scattering
        batch_size = batch_dict['batch_size']
        
        # 결과 담을 텐서 (B, 64, H, W)
        batch_fg_features = torch.zeros(
            (batch_size, self.fg_feature_dim, self.ny, self.nx),
            dtype=torch.float32, device=device
        )

        for b_idx in range(batch_size):
            pcd = foreground_pcd_list[b_idx]
            points_np = np.asarray(pcd.points)
            
            if points_np.shape[0] == 0:
                continue
            
            # Intensity 복원
            if pcd.has_colors():
                intensities_np = np.asarray(pcd.colors)[:, 0:1] 
            else:
                intensities_np = np.zeros((points_np.shape[0], 1))

            # Tensor 변환 (N, 4): x, y, z, i
            points_tensor = torch.from_numpy(np.hstack([points_np, intensities_np])).float().to(device)
            
            # Grid Coords 계산
            x_coords = ((points_tensor[:, 0] - self.point_cloud_range[0]) / self.voxel_x).floor().long()
            y_coords = ((points_tensor[:, 1] - self.point_cloud_range[1]) / self.voxel_y).floor().long()
            
            # 범위 밖 필터링
            mask_grid = (x_coords >= 0) & (x_coords < self.nx) & (y_coords >= 0) & (y_coords < self.ny)
            if not mask_grid.any():
                continue

            points_tensor = points_tensor[mask_grid]
            x_coords = x_coords[mask_grid]
            y_coords = y_coords[mask_grid]
            
            # --- 10D Feature Construction (PointPillars Style) ---
            # 1. (x, y, z, i) : points_tensor
            # 2. (x_c, y_c, z_c) : Offset from Cluster Mean
            #    점 1개가 곧 1개의 Pillar이므로, Mean = 점 자신. 따라서 Offset = 0
            zeros = torch.zeros((points_tensor.shape[0], 3), device=device)
            
            # 3. (x_p, y_p, z_p) : Offset from Pillar(Voxel) Center
            pillar_center_x = (x_coords.float() * self.voxel_x + self.x_offset)
            pillar_center_y = (y_coords.float() * self.voxel_y + self.y_offset)
            pillar_center_z = self.z_offset # 전체 기둥 중심
            
            offset_center = points_tensor[:, :3] - torch.stack(
                [pillar_center_x, pillar_center_y, torch.full_like(pillar_center_x, pillar_center_z)], 
                dim=1
            )
            
            # Feature 조립: [x,y,z,i,  0,0,0,  xc,yc,zc] -> Total 10 dim
            # distance가 필요한 경우 11이 될 수 있으나, 보통 PointPillar는 10을 씀.
            # (코드 상단 VFE에서 with_distance=False면 9개일 수 있음, 여기선 10개로 가정하고 거리 추가여부 확인)
            
            feature_list = [points_tensor, zeros, offset_center]
            
            # with_distance 옵션 대응 (필요시)
            if self.with_distance: # 보통 False로 둠
                 points_dist = torch.norm(points_tensor[:, :3], 2, 1, keepdim=True)
                 feature_list.append(points_dist)

            fg_features = torch.cat(feature_list, dim=1) # (N_points, 10)
            
            # --- Single MLP Processing ---
            # PFNLayer expects input shape: (NumPillars, NumPointsPerPillar, Channels)
            # Here: (N_points, 1, 10)
            fg_features = fg_features.unsqueeze(1) 
            
            # Linear(10->64) -> BN -> ReLU -> MaxPool
            # Output: (N_points, 1, 64) -> (N_points, 64) (after squeeze internally in pfn if needed, or here)
            fg_features = self.fg_pfn(fg_features) # Returns (N, 1, 64) because last_layer=True returns x_max
            fg_features = fg_features.squeeze(1)   # (N, 64)

            # --- Scattering (Max Pooling to Grid) ---
            indices = y_coords * self.nx + x_coords
            indices_expand = indices.view(-1, 1).expand(-1, self.fg_feature_dim)
            
            flat_bev = torch.zeros((self.ny * self.nx, self.fg_feature_dim), device=device)
            
            # Scatter Max
            flat_bev.scatter_(0, indices_expand, fg_features)
            
            # Reshape
            bev_map = flat_bev.view(self.ny, self.nx, self.fg_feature_dim).permute(2, 0, 1)
            batch_fg_features[b_idx] = bev_map

        batch_dict['foreground_bev_mask'] = batch_fg_features

        return batch_dict
    

class PillarVFE_forground_extraction_11D(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        
        # 10D 피처 계산 (x, y, z, i, x_c, y_c, z_c, x_p, y_p)
        # 4(xyz, i) + 3(f_cluster) + 3(f_center) = 10D
        num_point_features += 6 if self.use_absolute_xyz else 3
        
        if self.with_distance:
            num_point_features += 1 # (선택) 11D
            
        # ▼▼▼ [수정됨] PFNLayer 제거 ▼▼▼
        # PFNLayer(PointNet)를 사용하지 않으므로, 
        # num_filters, pfn_layers 관련 코드를 모두 제거합니다.
        
        # self.num_filters = self.model_cfg.NUM_FILTERS ... (제거)
        # num_filters = [num_point_features] + ... (제거)
        # pfn_layers = [] ... (제거)
        # self.pfn_layers = nn.ModuleList(pfn_layers) (제거)
        
        # 출력 차원을 저장 (PFN이 없으므로 10D 또는 11D가 됨)
        self.num_point_features = num_point_features 
        # ▲▲▲
        
        self.point_cloud_range = point_cloud_range
        
        self.voxel_x = voxel_size[0] # 0.16
        self.voxel_y = voxel_size[1] # 0.16
        self.voxel_z = voxel_size[2] # 4
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0] #  -> 0.08
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1] #  -> -39.6
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2] #  -> -1

        self.octree_voxel_size = 0.01 # 0.01 
        
    def get_output_feature_dim(self):
        return self.num_point_features
        # ▲▲▲
    
    # (get_paddings_indicator 함수는 동일 - 수정 없음)
    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1) 
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator
    
    # (extract_foreground_voxelgrid_batch 함수는 동일 - 수정 없음)
    def extract_foreground_voxelgrid_batch(self, background_pointcloud, current_pointcloud, voxel_size):
        if background_pointcloud.is_cuda:
            background_pointcloud = background_pointcloud.cpu()
        if current_pointcloud.is_cuda:
            current_pointcloud = current_pointcloud.cpu()
        batch_indices = torch.unique(current_pointcloud[:, 0]).int()
        foreground_pcd_list = []

        for b_idx in batch_indices:
            bg_mask = (background_pointcloud[:, 0] == b_idx)
            curr_mask = (current_pointcloud[:, 0] == b_idx)
            bg_points_np = background_pointcloud[bg_mask, 1:4].numpy()
            curr_batch_data = current_pointcloud[curr_mask]
            curr_points_np = curr_batch_data[:, 1:4].numpy()
            curr_batch_data_np = curr_batch_data.numpy()

            if curr_points_np.shape[0] == 0:
                foreground_pcd_list.append(o3d.geometry.PointCloud())
                continue
            
            pcd_bg = o3d.geometry.PointCloud()
            pcd_bg.points = o3d.utility.Vector3dVector(bg_points_np)
            
            if bg_points_np.shape[0] == 0:
                pcd_fg = o3d.geometry.PointCloud()
                pcd_fg.points = o3d.utility.Vector3dVector(curr_points_np)
                intensity_colors = np.clip(curr_batch_data_np[:, 4], 0, 1)
                pcd_fg.colors = o3d.utility.Vector3dVector(
                    np.stack([intensity_colors]*3, axis=-1)
                )
                foreground_pcd_list.append(pcd_fg)
                continue

            voxel_bg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_bg, voxel_size)
            bg_voxel_indices = {tuple(v.grid_index) for v in voxel_bg.get_voxels()}
            origin = voxel_bg.origin

            foreground_point_indices = []
            for i in range(curr_points_np.shape[0]):
                point = curr_points_np[i]
                grid_index = tuple(np.floor((point - origin) / voxel_size).astype(int))
                if grid_index not in bg_voxel_indices:
                    foreground_point_indices.append(i)
            
            if foreground_point_indices:
                fg_points_data = curr_batch_data_np[foreground_point_indices]
                pcd_fg = o3d.geometry.PointCloud()
                pcd_fg.points = o3d.utility.Vector3dVector(fg_points_data[:, 1:4])
                intensity_colors = np.clip(fg_points_data[:, 4], 0, 1)
                pcd_fg.colors = o3d.utility.Vector3dVector(
                    np.stack([intensity_colors]*3, axis=-1)
                )
                foreground_pcd_list.append(pcd_fg)
            else:
                foreground_pcd_list.append(o3d.geometry.PointCloud())
        return foreground_pcd_list

    def create_bev_mask_batch(self, foreground_pcd_list: List[o3d.geometry.PointCloud], 
                              batch_size: int, device: torch.device):
        pc_range = self.point_cloud_range
        min_x, min_y = pc_range[0], pc_range[1]
        size_x, size_y = self.voxel_x, self.voxel_y
        grid_size_x = int(np.round((pc_range[3] - pc_range[0]) / size_x)) 
        grid_size_y = int(np.round((pc_range[4] - pc_range[1]) / size_y)) 
        
        output_mask_tensor = torch.zeros((batch_size, 1, grid_size_x, grid_size_y), 
                                         dtype=torch.float32, device=device)
        
        for i in range(batch_size):
            fg_points_np = np.asarray(foreground_pcd_list[i].points)
            if fg_points_np.shape[0] == 0:
                continue
            
            x_indices = np.floor((fg_points_np[:, 0] - min_x) / size_x).astype(int)
            y_indices = np.floor((fg_points_np[:, 1] - min_y) / size_y).astype(int)
            
            valid_mask = (x_indices >= 0) & (x_indices < grid_size_x) & \
                         (y_indices >= 0) & (y_indices < grid_size_y)
            
            if not np.any(valid_mask):
                continue

            x_indices_valid = x_indices[valid_mask]
            y_indices_valid = y_indices[valid_mask]
            
            x_indices_torch = torch.from_numpy(x_indices_valid).long().to(device)
            y_indices_torch = torch.from_numpy(y_indices_valid).long().to(device)
            
            output_mask_tensor[i, 0, x_indices_torch, y_indices_torch] = 1.0
            
        return output_mask_tensor
    
    def forward(self, batch_dict, **kwargs):
        
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']

        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean 
        
        f_center = torch.zeros_like(voxel_features[:, :, :3]) 
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
            
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
            
        features = torch.cat(features, dim=-1) # (P, N, 10)

        voxel_count = features.shape[1]
        
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        
        features *= mask 
        
        # 2. 전경 BEV 마스크 생성 (B, 1, H, W)
        background_pointcloud = batch_dict['background']
        current_pointcloud = batch_dict['points'] 

        foreground_results = self.extract_foreground_voxelgrid_batch(
            background_pointcloud, 
            current_pointcloud, 
            voxel_size=self.octree_voxel_size
        )

        foreground_bev_mask = self.create_bev_mask_batch(
            foreground_results,
            batch_dict['batch_size'],
            features.device # 텐서 디바이스 일치
        )

        # mask = (background_pointcloud[:, 0] == 0)
        # batch_0_points = background_pointcloud[mask] # B, x, y, z, i
        # print(f"\n배치 0 (전경) 포인트 개수: {len(foreground_results[0].points)}")
        # if len(foreground_results[0].points) > 0:
        #     # (시각화) 첫 번째 배치의 전경(박스)을 빨간색으로, 배경을 회색으로 표시
        #     pcd_b0_bg = o3d.geometry.PointCloud()
        #     pcd_b0_bg.points = o3d.utility.Vector3dVector(batch_0_points.cpu().numpy()[:,1:4])
        #     pcd_b0_bg.paint_uniform_color([0.5, 0.5, 0.5]) # 회색
            
        #     # 전경은 반사도(회색조) 대신 빨간색으로 칠해서 확인
        #     foreground_results[0].paint_uniform_color([1.0, 0, 0]) # 빨간색
            
        #     o3d.visualization.draw_geometries(
        #         [foreground_results[0], pcd_b0_bg], 
        #         window_name="Batch 0 - Foreground (Red) vs Background (Gray)"
        #     )
        # 10D 피처(P, N, 10) -> (P, 10)

        pillar_features = features.max(dim=1, keepdim=False)[0] # Maximum value for each channel
        
        # 3. 결과 저장
        batch_dict['pillar_features'] = pillar_features     # (P, 10)
        batch_dict['foreground_bev_mask'] = foreground_bev_mask # (B, 1, H, W)
        
        return batch_dict