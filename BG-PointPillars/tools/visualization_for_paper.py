import argparse
import glob
from pathlib import Path

# try:
#     import open3d
#     from visual_utils import open3d_vis_utils as V
#     OPEN3D_FLAG = True
# except:
import mayavi.mlab as mlab
from visual_utils import visualize_utils as V
OPEN3D_FLAG = False

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from pcdet.models.detectors.detector3d_template import Detector3DTemplate
from pcdet.models.dense_heads.anchor_head_single import AnchorHeadSingle,AnchorHeadSingle_test
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate, build_dataloader,build_dataloader_for_random_sample
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

class pruned_Model(Detector3DTemplate):
    def __init__(self, original_model,model_cfg, num_class, dataset):
        super(pruned_Model, self).__init__(model_cfg, num_class, dataset)
        self.vfe = original_model.vfe
        self.map_to_bev_module = original_model.map_to_bev_module
        self.backbone_2d = original_model.backbone_2d
        self.dense_head = original_model.dense_head
        
    def forward(self, x):
        x = self.vfe(x)
        x = self.map_to_bev_module(x)
        x = self.backbone_2d(x)
        x = self.dense_head(x)
        pred_dicts, recall_dicts = self.post_processing(x)
        
        return pred_dicts, recall_dicts

# class pruned_Model(nn.Module):
#     def __init__(self, original_model):
#         super(pruned_Model, self).__init__()
#         self.vfe = original_model.vfe
#         self.map_to_bev_module = original_model.map_to_bev_module
#         self.backbone_2d = original_model.backbone_2d
#         # self.dense_head = original_model.dense_head
        
#     def forward(self, x):
#         x = self.vfe(x)
#         x = self.map_to_bev_module(x)
#         x = self.backbone_2d(x)
#         # x = self.dense_head(x)
        
#         return x

class pruned_Model_full(Detector3DTemplate):
    def __init__(self, original_model,input_channels,model_cfg, num_class, dataset):
        super(pruned_Model_full, self).__init__(model_cfg, num_class, dataset)
        self.vfe = original_model.vfe
        self.map_to_bev_module = original_model.map_to_bev_module
        self.backbone_2d = original_model.backbone_2d
        self.dense_head = AnchorHeadSingle_test(model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=input_channels,
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=np.array([608,608,1]),
            # point_cloud_range=np.array([0, -50.0, -10.0, 100.0, 50.0, -2.56]),
            point_cloud_range=np.array([0, -50.0, -8.0, 100.0, 50.0, -1.0]),
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            # voxel_size=[0.16447, 0.16447, 7.44]
            voxel_size=[0.16447, 0.16447, 7.0]
        )

    def forward(self, x):
        x = self.vfe(x)
        x = self.map_to_bev_module(x)
        x = self.backbone_2d(x)  
        x = self.dense_head(x)
        
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(x)
            return pred_dicts, recall_dicts
        
    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict  
      
class background_mini_pointnet(Detector3DTemplate):
    
    def __init__(self, original_model,model_cfg, num_class, dataset):
        super(background_mini_pointnet, self).__init__(model_cfg, num_class, dataset)
        self.vfe = original_model.vfe
        self.map_to_bev_module = original_model.map_to_bev_module
        self.backbone_2d = original_model.backbone_2d
        self.dense_head = original_model.dense_head
        
    def forward(self, x):
        x = self.vfe(x)
        x = self.map_to_bev_module(x)
        x = self.backbone_2d(x)
        x = self.dense_head(x)
        
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(x)
            return pred_dicts, recall_dicts
        
    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
    
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    
    # parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/A9_models/pointpillar_new_1_v2_dual_pointnet_background.yaml', help='specify the config for training')
    parser.add_argument('--cfg_file', type=str, default='/home/ivpg/PJW/CUDA-PointPillars/OpenPCDet/tools/cfgs/A9_models/pointpillar_new_1_v2_dual_pointnet_background_0.005_8_2nd.yaml', help='specify the config for training')
    # parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/RC_models/pointpillar_retrain.yaml', help='specify the config for training')
    # parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/RC_models/pointpillar_dual_pointnet_background.yaml', help='specify the config for training')
    
    # parser.add_argument('--ckpt', type=str, default='/mnt/hdd/pointpillar_A9_dataset_checkpoints/output/tools/cfgs/RC_models/pointpillar_dual_pointnet_background_retrain/default/4th_retrain/checkpoint_epoch_200.pth', help='checkpoint to start from')
    # parser.add_argument('--ckpt', type=str, default='/mnt/hdd/pointpillar_A9_dataset_checkpoints/output/tools/cfgs/RC_models/pointpillar_mini_pointnet_frozen_background_mean/default/ckpt/checkpoint_epoch_180.pth', help='checkpoint to start from')
    # parser.add_argument('--ckpt', type=str, default='/mnt/hdd/pointpillar_A9_dataset_checkpoints/output/tools/cfgs/A9_models/pointpillar_new_1_v2_retrain/default/5th_retrain/checkpoint_epoch_200.pth', help='checkpoint to start from')
    # parser.add_argument('--ckpt', type=str, default='/mnt/hdd/pointpillar_A9_dataset_checkpoints/output/tools/cfgs/A9_models/pointpillar_new_1_v2_dual_pointnet_background/default/revise_version/checkpoint_epoch_175.pth', help='checkpoint to start from')
    # parser.add_argument('--ckpt', type=str, default='/mnt/hdd/pointpillar_A9_dataset_checkpoints/output/tools/cfgs/RC_models/pointpillar_retrain/default/3rd_retrain/checkpoint_epoch_180.pth', help='checkpoint to start from')
    parser.add_argument('--ckpt', type=str, default='/mnt/8TB/PJW/BG-paper_weight/PointPillars/A9/BG_pointpillars_best_weight_a9.pth', help='checkpoint to start from')
    
    
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def draw_valid_region(valid_region = None):
    grid_size = 608
    grid = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

    # 랜덤하게 10,000개의 셀 선택
    num_cells_to_color = 10000
    random_indices = np.random.choice(grid_size * grid_size, num_cells_to_color, replace=False)

    # 빨간색으로 채우기
    for index in random_indices:
        x = index // grid_size
        y = index % grid_size
        grid[x, y] = [255, 0, 0]  # 빨간색

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid, interpolation='none')

    # 그리드 셀 경계선 추가
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    plt.title('608 x 608 Grid with 10000 Random Red Cells')
    # plt.axis('off')  # 축 숨기기
    plt.show()
    
def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    
    test_set, test_loader, sampler = build_dataloader_for_random_sample(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=logger, training=False
    )
    
    # test_set, test_loader, sampler = build_dataloader(
    #     dataset_cfg=cfg.DATA_CONFIG,
    #     class_names=cfg.CLASS_NAMES,
    #     batch_size=args.batch_size,
    #     dist=False, workers=args.workers, logger=logger, training=False
    # )
    
    # test_set, test_loader, train_sampler = build_dataloader(
    #     dataset_cfg=cfg.DATA_CONFIG,
    #     class_names=cfg.CLASS_NAMES,
    #     batch_size=args.batch_size,
    #     workers=args.workers,
    #     logger=logger,
    #     training=True,
    #     dist=False,
    #     merge_all_iters_to_one_epoch=True,
    #     total_epochs=1
    # )

    logger.info(f'Total number of samples: \t{len(test_set)}')

    # model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    # model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    
    ckpt_ = torch.load(args.ckpt,map_location="cuda:0")
    model = ckpt_['model'].float()
    print("model is successfully loaded!!")
    print(model)
    model = pruned_Model(model,model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.cuda()
    model.eval()
    
    with torch.no_grad():
        for idx, data_dict in enumerate(test_loader):
            # logger.info(f'Visualized sample index: \t{idx + 1}')
            
            # data_dict = test_loader.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            print(data_dict['frame_id'][0])
            # if data_dict['frame_id'][0] != "1651673120_856140376_s110_lidar_ouster_south":
            #     continue
            # draw_valid_region()
            
            ############
            
            # pred_dicts = []
            # temp = {}
            # temp['pred_boxes'] = torch.tensor(np.load(os.path.join("/home/jiwon/3d_object_detection/Complex-YOLO3D/src/complex_yolov7_3D_a9_v2_results_openpcdet_inference",data_dict['frame_id'][0]+"_pred_boxes.npy"))).cuda()
            # temp['pred_scores'] = torch.tensor(np.load(os.path.join("/home/jiwon/3d_object_detection/Complex-YOLO3D/src/complex_yolov7_3D_a9_v2_results_openpcdet_inference",data_dict['frame_id'][0]+"_pred_scores.npy"))).cuda()
            # temp['pred_labels'] = torch.tensor(np.load(os.path.join("/home/jiwon/3d_object_detection/Complex-YOLO3D/src/complex_yolov7_3D_a9_v2_results_openpcdet_inference",data_dict['frame_id'][0]+"_pred_labels.npy")).astype(np.int32)).cuda()
            # pred_dicts.append(temp)
            
            
            ###############3
            gt_label = data_dict['gt_boxes'][0][:,-1]
            
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     gt_boxes = data_dict['gt_boxes'][0],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
            #     valid_region=data_dict['voxel_coords']
            # )

            # if not OPEN3D_FLAG:
            #     mlab.show(stop=True)
            # if (torch.any(pred_dicts[0]['pred_labels']==3) or torch.any(gt_label==3.0)):
            # if data_dict['frame_id'][0][-3:] == "117":
            # if "1693205041.073476" in data_dict['frame_id'][0]:
            # if "1651673122_360563421" in data_dict['frame_id'][0]:
            # if "1693908985.384403_139" in data_dict['frame_id'][0]: #  1693908986.683251
                # draw_valid_region()

            print("Predict: ")
            print(len(pred_dicts[0]['pred_boxes']))
            print("GT: ")
            print(len(data_dict['gt_boxes'][0]))

            V.draw_scenes_for_paper(
                points=data_dict['points'][:, 1:], 
                # points=data_dict['background'][:, 1:], 
                ref_boxes=pred_dicts[0]['pred_boxes'],
                gt_boxes = data_dict['gt_boxes'][0],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                valid_region= data_dict['voxel_coords']
            )

            if not OPEN3D_FLAG:
                # mlab.axes(xlabel='X', ylabel='Y', zlabel='Z', line_width=2)
                mlab.show(stop=True)
            else:
                    continue
            
    logger.info('Demo done.')


if __name__ == '__main__':
    main()
