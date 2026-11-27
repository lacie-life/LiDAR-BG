import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
# import config.ComplexYolo3D_PointPillars_config as cnf
import config.ComplexYolo3D_PointPillars_config_withrobot as cnf

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []

        batch_size = coords[:, 0].max().int().item() + 1

        self.num_bev_features = pillar_features.shape[1]
        
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
            

        # batch_spatial_features_raw = batch_spatial_features.detach().cpu().numpy().astype(np.float32)
        # batch_spatial_features_raw = batch_spatial_features_raw.transpose(0,2,3,1) # NCHW -> NHWC
        #batch_spatial_features_raw.tofile(os.path.join("/media/ivpg/8TB/PJW/snpe_228/pointpillar_6490_a9_v2/backbone_train_histogram_QAT_input_norm", "backbone_input_"+batch_dict["frame_id"][0]+".raw"))
        
        # f = open("./backbone_train_list.txt",'a')
        # f.write(os.path.join("/workspace/pointpillar_6490/backbone_train", "backbone_input_"+batch_dict["frame_id"][0]+".raw"))
        # f.write("\n")
        
        return batch_dict

    
class makeComplexYOLO(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

        # self.BEV_HEIGHT = grid_size
        # self.BEV_WIDTH = grid_size

        self.minX = cnf.boundary['minX']
        self.maxX = cnf.boundary['maxX']
        self.minY = cnf.boundary['minY']
        self.maxY = cnf.boundary['maxY']
        self.minZ = cnf.boundary['minZ']
        self.maxZ = cnf.boundary['maxZ']

        self.bc = cnf.boundary

        self.Discretization = cnf.DISCRETIZATION

    def forward(self, batch_dict, **kwargs):

        coords = batch_dict['voxel_coords']

        batch_size = coords[:, 0].max().int().item() + 1
        
        # image_path = batch_dict['frame_id']

        ### make BEV Feature function ###
        comp_pseudo_image = []

        for batch_idx in range(batch_size):

            batch_mask = batch_dict['points'][:, 0] == batch_idx
            PointCloud = batch_dict['points'][batch_mask, 1:].cpu().numpy()

            mask_p = np.where((PointCloud[:, 2] >= self.minZ) & (PointCloud[:, 2] <= self.maxZ))

            PointCloud = PointCloud[mask_p]
            PointCloud_coor = PointCloud[:, :2]
            
            PointCloud[:, 2] = PointCloud[:, 2] - self.minZ # z축을 0m~ maxZ-minZ m 인 양수로 만듦

            ########################makeBVFeature##########################################################
            Height = cnf.BEV_HEIGHT + 1 # 608 + 1 = 609
            Width = cnf.BEV_WIDTH + 1 # 608 + 1 = 609

            PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / self.Discretization))
            PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / self.Discretization) + Width / 2)

            # sort-3times
            indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
            PointCloud = PointCloud[indices]
            PointCloud_coor = PointCloud_coor[indices]

            # Height Map
            heightMap = np.zeros((Height, Width))
            _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
            PointCloud_frac = PointCloud[indices]
            max_height = float(np.abs(self.maxZ - self.minZ))
            heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = (PointCloud_frac[:, 2]) / max_height #(PointCloud_frac[:, 2]  + self.minZ) / max_height
        
            # Intensity Map & DensityMap
            intensityMap = np.zeros((Height, Width))
            densityMap = np.zeros((Height, Width))

            _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
            PointCloud_top = PointCloud[indices]

            normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))

            intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3] 
            densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

            RGB_Map = np.zeros((3, Height - 1, Width - 1))
            RGB_Map[2, :, :] = np.flip(np.rot90(densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH], 1), 0)  # r_map
            RGB_Map[1, :, :] = np.flip(np.rot90(heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH], 1), 0)  # g_map
            RGB_Map[0, :, :] = np.flip(np.rot90(intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH], 1), 0)  # b_map

            comp_pseudo_image.append(RGB_Map)
        comp_pseudo_image = torch.tensor(np.array(comp_pseudo_image)).to(batch_dict['points'].device)

        ########################makeBVFeature##########################################################
        batch_dict['spatial_features'] = comp_pseudo_image.float() # 3ch pseudo-image

        return batch_dict
    

class makeComplexYOLO_v2(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

        # self.BEV_HEIGHT = grid_size
        # self.BEV_WIDTH = grid_size

        self.minX = cnf.boundary['minX']
        self.maxX = cnf.boundary['maxX']
        self.minY = cnf.boundary['minY']
        self.maxY = cnf.boundary['maxY']
        self.minZ = cnf.boundary['minZ']
        self.maxZ = cnf.boundary['maxZ']

        self.bc = cnf.boundary

        self.Discretization = cnf.DISCRETIZATION

    def forward(self, batch_dict, **kwargs):

        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']

        batch_size = coords[:, 0].max().int().item() + 1
        
        image_path = batch_dict['frame_id']

        ### make BEV Feature function ###
        comp_pseudo_image = []

        for batch_idx in range(batch_size):

            spatial_feature = torch.zeros(
                1,
                self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars

            avg_heightMap = spatial_feature.view(1, self.ny, self.nx).cpu().numpy()

            # print(avg_height.shape)

            batch_mask = batch_dict['points'][:, 0] == batch_idx
            PointCloud = batch_dict['points'][batch_mask, 1:].cpu().numpy()

            mask_p = np.where((PointCloud[:, 2] >= self.minZ) & (PointCloud[:, 2] <= self.maxZ))

            PointCloud = PointCloud[mask_p]
            PointCloud_coor = PointCloud[:, :2]
            
            PointCloud[:, 2] = PointCloud[:, 2] - self.minZ # z축을 0m~ maxZ-minZ m 인 양수로 만듦

            ########################makeBVFeature##########################################################
            Height = cnf.BEV_HEIGHT + 1 # 608 + 1 = 609
            Width = cnf.BEV_WIDTH + 1 # 608 + 1 = 609

            PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / self.Discretization))
            PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / self.Discretization) + Width / 2)

            # sort-3times
            indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
            PointCloud = PointCloud[indices]
            PointCloud_coor = PointCloud_coor[indices]

            # Height Map
            heightMap = np.zeros((Height, Width))
            _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
            PointCloud_frac = PointCloud[indices]
            max_height = float(np.abs(self.maxZ - self.minZ))
            heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = (PointCloud_frac[:, 2]) / max_height #(PointCloud_frac[:, 2]  + self.minZ) / max_height
            avg_heightMap = avg_heightMap / max_height
        
            # Intensity Map & DensityMap
            intensityMap = np.zeros((Height, Width))
            densityMap = np.zeros((Height, Width))

            _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
            PointCloud_top = PointCloud[indices]

            normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))

            intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3] 
            densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

            RGB_Map = np.zeros((4, Height - 1, Width - 1))
            RGB_Map[3, :, :] = avg_heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]
            RGB_Map[2, :, :] = np.flip(np.rot90(densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH], 1), 0)  # r_map
            RGB_Map[1, :, :] = np.flip(np.rot90(heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH], 1), 0)  # g_map
            RGB_Map[0, :, :] = np.flip(np.rot90(intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH], 1), 0)  # b_map

            # non_anign_pixel = 0
            # for i in range(608):
            #     for j in range(608):
            #         if (RGB_Map[1, i, j] != 0 and RGB_Map[3, i, j] != 0) or (RGB_Map[1, i, j] == 0 and RGB_Map[3, i, j] == 0):
            #             non_anign_pixel = non_anign_pixel + 0
            #         else:
            #             non_anign_pixel = non_anign_pixel + 1

            # print(non_anign_pixel)

            # # non_zero_mask = np.any(RGB_Map != 0, axis = 0) # non zero location
            # zero_mask = np.all(RGB_Map == 0, axis = 0) # zero location
            # # mean_value = np.mean(RGB_Map[:,non_zero_mask], axis = 1).reshape(-1,1) # non zero location 채널 별 평균
            # RGB_Map[:,zero_mask] += 0.5 # 채널 별로 zero location에 평균 더해주기

            # RGB_Map = np.concatenate([RGB_Map, pillar_center_xy_flip],axis = 0)
            # cv2.imwrite("/media/ivpg/8TB/LSH/pointpillar_a9/temp/densityMap.jpg",RGB_Map[2,:,:]*255)
            # cv2.imwrite("/media/ivpg/8TB/LSH/pointpillar_a9/temp/heightMap.jpg",RGB_Map[1,:,:]*255)
            # cv2.imwrite("/media/ivpg/8TB/LSH/pointpillar_a9/temp/intensityMap.jpg",RGB_Map[0,:,:]*255)

            # cv2.imwrite("/media/ivpg/8TB/LSH/pointpillar_a9/temp/RGB_MAP.jpg",RGB_Map[:,:,:]*10000)

            # image = RGB_Map[0:3, :, :]*255
            # image = image.transpose(1, 2, 0).astype(np.uint8)
            # image = np.ascontiguousarray(image).astype(np.uint8)

            # image_c = RGB_Map[3, :, :]*255
            # # image_c = image_c.transpose(1, 0).astype(np.uint8)
            # image_c = np.ascontiguousarray(image_c).astype(np.uint8)


            # # # print(len(bbox_gt[:,0]))

            # for i in range(len(bbox_gt)):

            #     # print(bbox_gt[i])

            #     # pt1 = (int((bbox_gt[i, 0]-bbox_gt[i, 3]/2)  / 100 * 608), int((bbox_gt[i, 1]-bbox_gt[i, 4]/2)  / 100 * 608))
            #     # pt2 = (int((bbox_gt[i, 0]+bbox_gt[i, 3]/2)  / 100 * 608), int((bbox_gt[i, 1]-bbox_gt[i, 4]/2)  / 100 * 608)) 
            #     # pt3 = (int((bbox_gt[i, 0]-bbox_gt[i, 3]/2)  / 100 * 608), int((bbox_gt[i, 1]+bbox_gt[i, 4]/2)  / 100 * 608)) 
            #     # pt4 = (int((bbox_gt[i, 0]+bbox_gt[i, 3]/2)  / 100 * 608), int((bbox_gt[i, 1]+bbox_gt[i, 4]/2)  / 100 * 608))

            #     x, y, z, dx, dy, dz, yaw = bbox_gt[i, :7]
            #     x = x.cpu().numpy()
            #     y = y.cpu().numpy()
            #     dx = dx.cpu().numpy()
            #     dy = dy.cpu().numpy()
                
            #     corners = np.array([
            #         [x - dx / 2, y - dy / 2], #pt1
            #         [x + dx / 2, y - dy / 2], #pt2
            #         [x + dx / 2, y + dy / 2], #pt4
            #         [x - dx / 2, y + dy / 2], #pt3
            #         [x - dx / 2, y - dy / 2]
            #     ])

            #     yaw = -yaw.cpu().numpy() 

            #     # Rotate the corners based on the yaw angle
            #     rotation_matrix = np.array([
            #         [np.cos(yaw), -np.sin(yaw)],
            #         [np.sin(yaw), np.cos(yaw)]
            #     ])
            #     corners = np.dot(corners - [x, y], rotation_matrix) + [x, y]

            #     # print("BEV corner: ", corners)

            #     corners[:, 0] = corners[:, 0] / self.Discretization
            #     corners[:, 1] = -corners[:, 1] / self.Discretization + Width/2


            #     # Draw lines between consecutive points in the corners array
            #     for i in range(len(corners) - 1):
            #         # Convert start and end points to integer tuples (required by OpenCV)
            #         pt1 = (int(corners[i, 0]), int(corners[i, 1]))  # Current point
            #         pt2 = (int(corners[i+1, 0]), int(corners[i+1, 1]))  # Next point
                    
            #         # Draw a red line (in OpenCV, red is BGR (0, 0, 255))
            #         cv2.line(image, pt1, pt2, (0, 0, 255), thickness=2)
            #         cv2.line(image_c, pt1, pt2, (0, 0, 255), thickness=2)

            #     # Connect the last point to the first to create a closed shape (if needed)
            #     if len(corners) > 2:  # Only if there are at least 3 points
            #         pt1 = (int(corners[-1, 0]), int(corners[-1, 1]))  # Last point
            #         pt2 = (int(corners[0, 0]), int(corners[0, 1]))  # First point
            #         cv2.line(image, pt1, pt2, (0, 0, 255), thickness=2)
            #         cv2.line(image_c, pt1, pt2, (0, 0, 255), thickness=2)

            # cv2.imwrite(f"/media/ivpg/8TB/LSH/pointpillar_a9/temp/RGB_MAP_{image_path}.jpg",image)
            # cv2.imwrite(f"/media/ivpg/8TB/LSH/pointpillar_a9/temp/RGB_MAP_{image_path}_c.jpg",image_c)
            # draw_bev_boxes(points_tmp, gt_boxes_tmp)

            comp_pseudo_image.append(RGB_Map)

        # comp_pseudo_image = torch.tensor(comp_pseudo_image).to(batch_dict['points'].device)
        comp_pseudo_image = torch.tensor(np.array(comp_pseudo_image)).to(batch_dict['points'].device)
        # print('min: ', comp_pseudo_image[0, 0, :, :].min())
        # print('max: ', comp_pseudo_image[0 ,0, :, :].max())
        # print('min: ', comp_pseudo_image[0, 1, :, :].min())
        # print('max: ', comp_pseudo_image[0, 1, :, :].max())
        # print('min: ', comp_pseudo_image[0 ,2, :, :].min())
        # print('max: ', comp_pseudo_image[0 ,2, :, :].max())
        # print('min: ', comp_pseudo_image[0 ,3, :, :].min())
        # print('max: ', comp_pseudo_image[0 ,3, :, :].max())
        # img_raw = np.array(comp_pseudo_image.cpu().permute(0, 2, 3, 1).squeeze(dim=0),dtype = np.float32)
        # f = open('/media/ivpg/8TB/LSH/pointpillar_a9/temp/A9_raw_train_v2_list.txt', 'a')
        # img_raw = np.array(comp_pseudo_image.cpu().permute(0, 2, 3, 1).squeeze(dim=0),dtype = np.float32)
        # img_raw.tofile(os.path.join(f'/media/ivpg/8TB/LSH/pointpillar_a9/temp/A9_raw_train_v2/RGB_MAP_{image_path}.raw'))
        # f.write(os.path.join(f'/data/local/tmp/LSH/A9_raw_train_v2/RGB_MAP_{image_path}.raw'))
        # f.write('\n')
        # f = open('/media/ivpg/8TB/LSH/pointpillar_a9/temp/A9_raw_test_v2_list.txt', 'a')
        # img_raw = np.array(comp_pseudo_image.cpu().permute(0, 2, 3, 1).squeeze(dim=0),dtype = np.float32)
        # img_raw.tofile(os.path.join(f'/media/ivpg/8TB/LSH/pointpillar_a9/temp/A9_raw_test_v2/RGB_MAP_{image_path}.raw'))
        # f.write(os.path.join(f'/data/local/tmp/LSH/A9_raw_test_v2/RGB_MAP_{image_path}.raw'))
        # f.write('\n')
        ########################makeBVFeature##########################################################
        batch_dict['spatial_features'] = comp_pseudo_image.float() # 3ch pseudo-image

        return batch_dict
    
class PointPillarScatter_skip(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1
        
    def forward(self, batch_dict, **kwargs):
        
        return batch_dict
    

class PointPillarScatter_onnx(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        
        spatial_feature = torch.zeros(
            self.num_bev_features,
            self.nz * self.nx * self.ny,
            dtype=pillar_features.dtype,
            device=pillar_features.device)

        indices = coords[:, 1] + coords[:, 2] * self.nx + coords[:, 3]
        indices = indices.type(torch.long)
        pillars = pillar_features.t()
        spatial_feature[:, indices] = pillars
        spatial_feature = spatial_feature.unsqueeze(dim = 0)
        spatial_feature = spatial_feature.view(1, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = spatial_feature
        
        
        return batch_dict


class PointPillarScatter_norm(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        
        # pillar_features_raw = np.array(pillar_features.cpu(),dtype = np.float32)
        # pillar_features_raw.tofile(os.path.join("/mnt/hdd/KITTI_dataset/pointpillar_preprocessing_python", "pillar_after_pointnet_"+batch_dict["frame_id"][0]+".raw"))
        
        batch_size = coords[:, 0].max().int().item() + 1

        self.num_bev_features = pillar_features.shape[1]
        
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        
        ############ for 0~1 normalization #############
        
        for i in range(batch_size):
            
            max_per_batch = torch.max(batch_spatial_features[i].view(self.num_bev_features,-1),dim = 1)[0].unsqueeze(1).unsqueeze(2)
            max_per_batch += 1e-6
            batch_spatial_features[i] /= max_per_batch
        
        # batch_spatial_features_raw = batch_spatial_features.detach().cpu().numpy().astype(np.float32)
        # batch_spatial_features_raw = batch_spatial_features_raw.transpose(0,2,3,1) # NCHW -> NHWC
        # batch_spatial_features_raw.tofile(os.path.join("/mnt/hdd/snpe_212/pointpillar_6490_a9_v2/backbone_train_norm", "backbone_input_"+batch_dict["frame_id"][0]+".raw"))
       
        ############ for 0~1 normalization #############       
        
        
        batch_dict['spatial_features'] = batch_spatial_features
        
        
        return batch_dict
    
class PointPillarScatter_mean_plus(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        
        # pillar_features_raw = np.array(pillar_features.cpu(),dtype = np.float32)
        # pillar_features_raw.tofile(os.path.join("/mnt/hdd/KITTI_dataset/pointpillar_preprocessing_python", "pillar_after_pointnet_"+batch_dict["frame_id"][0]+".raw"))
        
        batch_size = coords[:, 0].max().int().item() + 1

        self.num_bev_features = pillar_features.shape[1]
        
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        
        ############ for 0~1 normalization #############
        
        for i in range(batch_size):
            
            non_zero_mask = torch.any(batch_spatial_features[i] != 0, dim = 0) # non zero location
            zero_mask = torch.all(batch_spatial_features[i] == 0, dim = 0) # zero location
            mean_value = torch.mean(batch_spatial_features[i][:,non_zero_mask], dim = 1).view(-1,1) # non zero location 채널 별 평균
            batch_spatial_features[i][:,zero_mask] += mean_value # 채널 별로 zero location에 평균 더해주기
            
            # print(batch_spatial_features[i][batch_spatial_features[i] == 0])
            
        
        # batch_spatial_features_raw = batch_spatial_features.detach().cpu().numpy().astype(np.float32)
        # batch_spatial_features_raw = batch_spatial_features_raw.transpose(0,2,3,1) # NCHW -> NHWC
        # batch_spatial_features_raw.tofile(os.path.join("/mnt/hdd/snpe_212/pointpillar_6490_a9_v2/backbone_train_mean_plus_retrain", "backbone_input_"+batch_dict["frame_id"][0]+".raw"))
       
        ############ for 0~1 normalization #############       
        
        
        batch_dict['spatial_features'] = batch_spatial_features
        
        
        return batch_dict
    
class PointPillarScatter_adding_background(nn.Module):
    
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        
        # pillar_features_raw = np.array(pillar_features.cpu(),dtype = np.float32)
        # pillar_features_raw.tofile(os.path.join("/mnt/hdd/KITTI_dataset/pointpillar_preprocessing_python", "pillar_after_pointnet_"+batch_dict["frame_id"][0]+".raw"))
        
        batch_size = coords[:, 0].max().int().item() + 1

        self.num_bev_features = pillar_features.shape[1]
        
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_spatial_features_add_background = torch.zeros((batch_size,self.num_bev_features * self.nz * 2,self.ny, self.nx)).cuda()
        
        # def visualize_feature_map_with_borders(feature_map):
        #     num_channels = feature_map.shape[0]
        #     num_cols = 8
        #     num_rows = num_channels // num_cols
        #     if num_channels % num_cols != 0:
        #         num_rows += 1
            
        #     fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, num_rows * 2))
            
        #     for i in range(num_channels):
        #         row = i // num_cols
        #         col = i % num_cols
        #         ax = axes[row, col]
        #         ax.imshow(feature_map[i])
        #         # 테두리를 추가하여 구분
        #         for spine in ax.spines.values():
        #             spine.set_visible(True)
        #             spine.set_edgecolor('white')  # 테두리 색상을 흰색으로 설정
        #             spine.set_linewidth(2)  # 테두리 두께를 2로 설정
        #         ax.axis('off')
            
        #     # 남은 서브플롯을 비활성화
        #     for j in range(i + 1, num_rows * num_cols):
        #         fig.delaxes(axes.flatten()[j])
            
        #     # 배경 색상을 검정색으로 설정
        #     fig.patch.set_facecolor('black')
            
        #     plt.tight_layout()
        #     plt.show()
        ############ for adding background #############
        
        for i in range(batch_size):
        #     max_per_batch = batch_spatial_features[i].view(-1).max().item()
        
            # if "south" in batch_dict["frame_id"][i]: 
            #     background = torch.tensor(np.load('./data/A9_dataset/training/background_a9/a9_south_background_pointpillar_mean.npy').reshape(self.ny, self.nx,self.num_bev_features * self.nz)).permute(2,0,1).cuda()
            # elif "north" in batch_dict["frame_id"][i]: 
            #     background = torch.tensor(np.load('./data/A9_dataset/training/background_a9/a9_north_background_pointpillar_mean.npy').reshape(self.ny, self.nx,self.num_bev_features * self.nz)).permute(2,0,1).cuda()
            
            if "south" in batch_dict["frame_id"][i]: 
                background = torch.tensor(np.load('./data/A9_dataset/training/background_a9/a9_old_south_background_pointpillar_mean.npy').reshape(self.ny, self.nx,self.num_bev_features * self.nz)).permute(2,0,1).cuda()
            elif "north" in batch_dict["frame_id"][i]: 
                background = torch.tensor(np.load('./data/A9_dataset/training/background_a9/a9_old_north_background_pointpillar_mean.npy').reshape(self.ny, self.nx,self.num_bev_features * self.nz)).permute(2,0,1).cuda()
            
            # visualize_feature_map_with_borders(background.detach().cpu().numpy().astype(np.float32))
            # visualize_feature_map_with_borders(batch_spatial_features[i].detach().cpu().numpy().astype(np.float32))
            
            temp = torch.cat((batch_spatial_features[i],background))
            batch_spatial_features_add_background[i] += temp
            
            # print(batch_spatial_features[i].view(-1).max().item())
        

        
        ############ for adding background #############       
        
        
        # batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['spatial_features'] = batch_spatial_features_add_background
        
        # batch_spatial_features_raw = batch_spatial_features.detach().cpu().numpy().astype(np.float32)
        # batch_spatial_features_raw = batch_spatial_features_raw.transpose(0,2,3,1) # NCHW -> NHWC
        # batch_spatial_features_raw.tofile(os.path.join("/mnt/hdd/snpe_212/pointpillar_6490_a9_v2/backbone_train", "backbone_input_"+batch_dict["frame_id"][0]+".raw"))
        # f = open("./backbone_train_list.txt",'a')
        # f.write(os.path.join("/workspace/pointpillar_6490/backbone_train", "backbone_input_"+batch_dict["frame_id"][0]+".raw"))
        # f.write("\n")
        
        # batch_spatial_features = np.fromfile(os.path.join("/mnt/hdd/KITTI_dataset/pointpillar_preprocessing_c++","after_scatter_"+batch_dict["frame_id"][0] + ".bin"),dtype = np.float32).reshape(1,64,496,432)
        # batch_spatial_features = torch.tensor(batch_spatial_features).to("cuda:0")        
        # batch_dict['spatial_features'] = batch_spatial_features
        
        return batch_dict

class PointPillarScatter_adding_background_RC(nn.Module):
    
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        
        batch_size = coords[:, 0].max().int().item() + 1

        self.num_bev_features = pillar_features.shape[1]
        
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_spatial_features_add_background = torch.zeros((batch_size,self.num_bev_features * self.nz * 2,self.ny, self.nx)).cuda()
        

        
        
        for i in range(batch_size):

            if "_117" in batch_dict["frame_id"][i]: 
                background = torch.tensor(np.load('./data/RC_dataset/training/background_RC/RC_117_background_pointpillar_mean.npy').reshape(self.ny, self.nx,self.num_bev_features * self.nz)).permute(2,0,1).cuda()
            elif "_120" in batch_dict["frame_id"][i]: 
                background = torch.tensor(np.load('./data/RC_dataset/training/background_RC/RC_120_background_pointpillar_mean.npy').reshape(self.ny, self.nx,self.num_bev_features * self.nz)).permute(2,0,1).cuda()
            elif "_136" in batch_dict["frame_id"][i]: 
                background = torch.tensor(np.load('./data/RC_dataset/training/background_RC/RC_136_background_pointpillar_mean.npy').reshape(self.ny, self.nx,self.num_bev_features * self.nz)).permute(2,0,1).cuda()
            elif "_139" in batch_dict["frame_id"][i]: 
                background = torch.tensor(np.load('./data/RC_dataset/training/background_RC/RC_139_background_pointpillar_mean.npy').reshape(self.ny, self.nx,self.num_bev_features * self.nz)).permute(2,0,1).cuda()
            
    
            
            temp = torch.cat((batch_spatial_features[i],background))
            batch_spatial_features_add_background[i] += temp

        batch_dict['spatial_features'] = batch_spatial_features_add_background
        
        
        return batch_dict

class PointPillarScatter_adding_background_RC_median(nn.Module):
    
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []

        batch_size = coords[:, 0].max().int().item() + 1

        self.num_bev_features = pillar_features.shape[1]
        
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_spatial_features_add_background = torch.zeros((batch_size,self.num_bev_features * self.nz * 2,self.ny, self.nx)).cuda()
        
        
        for i in range(batch_size):

            if "_117" in batch_dict["frame_id"][i]: 
                background = torch.tensor(np.load('./data/RC_dataset/training/background_RC/RC_117_background_pointpillar_median.npy').reshape(self.ny, self.nx,self.num_bev_features * self.nz)).permute(2,0,1).cuda()
            elif "_120" in batch_dict["frame_id"][i]: 
                background = torch.tensor(np.load('./data/RC_dataset/training/background_RC/RC_120_background_pointpillar_median.npy').reshape(self.ny, self.nx,self.num_bev_features * self.nz)).permute(2,0,1).cuda()
            elif "_136" in batch_dict["frame_id"][i]: 
                background = torch.tensor(np.load('./data/RC_dataset/training/background_RC/RC_136_background_pointpillar_median.npy').reshape(self.ny, self.nx,self.num_bev_features * self.nz)).permute(2,0,1).cuda()
            elif "_139" in batch_dict["frame_id"][i]: 
                background = torch.tensor(np.load('./data/RC_dataset/training/background_RC/RC_139_background_pointpillar_median.npy').reshape(self.ny, self.nx,self.num_bev_features * self.nz)).permute(2,0,1).cuda()
            temp = torch.cat((batch_spatial_features[i],background))
            batch_spatial_features_add_background[i] += temp
        
 
        batch_dict['spatial_features'] = batch_spatial_features_add_background
        
        
        return batch_dict
    
class PointPillarScatter_adding_background_median(nn.Module):
    
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        
        # pillar_features_raw = np.array(pillar_features.cpu(),dtype = np.float32)
        # pillar_features_raw.tofile(os.path.join("/mnt/hdd/KITTI_dataset/pointpillar_preprocessing_python", "pillar_after_pointnet_"+batch_dict["frame_id"][0]+".raw"))
        
        batch_size = coords[:, 0].max().int().item() + 1

        self.num_bev_features = pillar_features.shape[1]
        
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_spatial_features_add_background = torch.zeros((batch_size,self.num_bev_features * self.nz * 2,self.ny, self.nx)).cuda()
        
        ############ for 0~1 normalization #############
        
        for i in range(batch_size):
        #     max_per_batch = batch_spatial_features[i].view(-1).max().item()
        
            if "south" in batch_dict["frame_id"][i]: 
                background = torch.tensor(np.load('./data/A9_dataset/training/background_a9/a9_south_background_pointpillar_median.npy').reshape(self.ny, self.nx,self.num_bev_features * self.nz)).permute(2,0,1).cuda()
            elif "north" in batch_dict["frame_id"][i]: 
                background = torch.tensor(np.load('./data/A9_dataset/training/background_a9/a9_north_background_pointpillar_median.npy').reshape(self.ny, self.nx,self.num_bev_features * self.nz)).permute(2,0,1).cuda()
            

            temp = torch.cat((batch_spatial_features[i],background))
            batch_spatial_features_add_background[i] += temp
            
            # print(batch_spatial_features[i].view(-1).max().item())
        
        
        
        ############ for 0~1 normalization #############       
        
        
        # batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['spatial_features'] = batch_spatial_features_add_background
        
        # batch_spatial_features_raw = batch_spatial_features.detach().cpu().numpy().astype(np.float32)
        # batch_spatial_features_raw = batch_spatial_features_raw.transpose(0,2,3,1) # NCHW -> NHWC
        # batch_spatial_features_raw.tofile(os.path.join("/mnt/hdd/snpe_212/pointpillar_6490_a9_v2/backbone_train", "backbone_input_"+batch_dict["frame_id"][0]+".raw"))
        # f = open("./backbone_train_list.txt",'a')
        # f.write(os.path.join("/workspace/pointpillar_6490/backbone_train", "backbone_input_"+batch_dict["frame_id"][0]+".raw"))
        # f.write("\n")
        
        # batch_spatial_features = np.fromfile(os.path.join("/mnt/hdd/KITTI_dataset/pointpillar_preprocessing_c++","after_scatter_"+batch_dict["frame_id"][0] + ".bin"),dtype = np.float32).reshape(1,64,496,432)
        # batch_spatial_features = torch.tensor(batch_spatial_features).to("cuda:0")        
        # batch_dict['spatial_features'] = batch_spatial_features
        
        return batch_dict


class PointPillarScatter_adding_background_dual_encoder(nn.Module):
    
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        print("!!!!!!!!!!!!!!!!!!!!!")
        print(grid_size)
        print("!!!!!!!!!!!!!!!!!!!!!")
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        
        # pillar_features_raw = np.array(pillar_features.cpu(),dtype = np.float32)
        # pillar_features_raw.tofile(os.path.join("/mnt/hdd/KITTI_dataset/pointpillar_preprocessing_python", "pillar_after_pointnet_"+batch_dict["frame_id"][0]+".raw"))
        
        batch_size = coords[:, 0].max().int().item() + 1

        self.num_bev_features = pillar_features.shape[1]
        
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
     
        ##### background psuedo image generate #####
            
        pillar_background_features, background_coords = batch_dict['pillar_background_features'], batch_dict['voxel_coords_background']
        batch_spatial_background_features = []
        
        for batch_idx in range(batch_size):
            
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_background_features.dtype,
                device=pillar_background_features.device)

            batch_mask = background_coords[:, 0] == batch_idx
            this_coords = background_coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_background_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_background_features.append(spatial_feature)

        batch_spatial_background_features = torch.stack(batch_spatial_background_features, 0)
        batch_spatial_background_features = batch_spatial_background_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        
        ##### background psuedo image generate #####
            
        batch_spatial_features_add_background = torch.cat([batch_spatial_features,batch_spatial_background_features],dim = 1)
        
        batch_dict['spatial_features'] = batch_spatial_features_add_background
        
        return batch_dict


class PointPillarScatter_adding_background_dual_encoder_pseudo_norm(nn.Module):
    
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        
        # pillar_features_raw = np.array(pillar_features.cpu(),dtype = np.float32)
        # pillar_features_raw.tofile(os.path.join("/mnt/hdd/KITTI_dataset/pointpillar_preprocessing_python", "pillar_after_pointnet_"+batch_dict["frame_id"][0]+".raw"))
        
        batch_size = coords[:, 0].max().int().item() + 1

        self.num_bev_features = pillar_features.shape[1]
        
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        
        for i in range(batch_size):
            
            max_per_batch = torch.max(batch_spatial_features[i].view(self.num_bev_features,-1),dim = 1)[0].unsqueeze(1).unsqueeze(2)
            max_per_batch += 1e-6
            batch_spatial_features[i] /= max_per_batch
            
        ##### background psuedo image generate #####
            
        pillar_background_features, background_coords = batch_dict['pillar_background_features'], batch_dict['voxel_coords_background']
        batch_spatial_background_features = []
        
        for batch_idx in range(batch_size):
            
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_background_features.dtype,
                device=pillar_background_features.device)

            batch_mask = background_coords[:, 0] == batch_idx
            this_coords = background_coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_background_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_background_features.append(spatial_feature)

        batch_spatial_background_features = torch.stack(batch_spatial_background_features, 0)
        batch_spatial_background_features = batch_spatial_background_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        
        for i in range(batch_size):
            
            max_per_batch = torch.max(batch_spatial_background_features[i].view(self.num_bev_features,-1),dim = 1)[0].unsqueeze(1).unsqueeze(2)
            max_per_batch += 1e-6
            batch_spatial_background_features[i] /= max_per_batch
            
        ##### background psuedo image generate #####
            
        batch_spatial_features_add_background = torch.cat([batch_spatial_features,batch_spatial_background_features],dim = 1)
        
        batch_dict['spatial_features'] = batch_spatial_features_add_background
        
        return batch_dict


class PointPillarScatter_adding_foreground_extraction(nn.Module):
    
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        
        # pillar_features_raw = np.array(pillar_features.cpu(),dtype = np.float32)
        # pillar_features_raw.tofile(os.path.join("/mnt/hdd/KITTI_dataset/pointpillar_preprocessing_python", "pillar_after_pointnet_"+batch_dict["frame_id"][0]+".raw"))
        
        batch_size = coords[:, 0].max().int().item() + 1

        self.num_bev_features = pillar_features.shape[1]
        
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        
        
        ##### foreground psuedo image generate #####
        batch_foreground_bev_features = batch_dict['foreground_bev_mask']

        # temp = batch_foreground_bev_features[0,:]
        # temp = temp.permute(1,2,0).cpu().numpy()
        # temp = batch_spatial_features[0,:]
        # temp = temp.permute(1,2,0).detach().cpu().numpy()
        # temp = temp[:,:,0]
        # cv2.imshow("ss",temp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        batch_spatial_features_add_foreground = torch.cat([batch_spatial_features,batch_foreground_bev_features],dim = 1)
        
        batch_dict['spatial_features'] = batch_spatial_features_add_foreground
        
        return batch_dict
    