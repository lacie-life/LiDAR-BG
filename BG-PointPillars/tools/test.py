import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import torch_pruning as tp
import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.models.detectors.detector3d_template import Detector3DTemplate
from pcdet.models.dense_heads.anchor_head_single import AnchorHeadSingle,AnchorHeadSingle_test

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
            point_cloud_range=np.array([0, -50.0, -10.0, 100.0, 50.0, -2.56]),
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            voxel_size=[0.16447, 0.16447, 7.44]
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
    
MAX_VOXELS = 20000
MAX_NUM_POINTS = 32

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/A9_models/pointpillar_new_1_v2_best.yaml', help='specify the config for training')
    # parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/withrobot_models/DHIAHpillars.yaml', help='specify the config for training')
    # parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/withrobot_models/pointpillar_label_in_camera_fov.yaml', help='specify the config for training')
    # parser.add_argument('--cfg_file', type=str, default='/home/ivpg/PJW/CUDA-PointPillars/OpenPCDet/tools/cfgs/A9_models/pointpillar_new_1_v2_best.yaml', help='specify the config for training')
    # /home/ivpg/PJW/CUDA-PointPillars/OpenPCDet/tools/cfgs/A9_models/pointpillar_new_1_v2_dual_pointnet_background_0.005_8_2nd.yaml
    parser.add_argument('--cfg_file', type=str, default='/home/ivpg/PJW/CUDA-PointPillars/OpenPCDet/tools/cfgs/A9_models/pointpillar_new_1_v2_best_2nd.yaml', help='specify the config for training')
    # parser.add_argument('--cfg_file', type=str, default='/home/ivpg/PJW/CUDA-PointPillars/OpenPCDet/tools/cfgs/A9_models/pointpillar_new_1_v2_dual_pointnet_background_0.005_8_2nd.yaml', help='specify the config for training')

    # parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/A9_models/pointpillar_new_1_v2_reretrain_0.001_8.yaml', help='specify the config for training')
    # parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/RC_models/pointpillar_foreground_extraction_octree_65D.yaml', help='specify the config for training')
    # parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/A9_models/pointpillar_old_no_aug.yaml', help='specify the config for training')
    
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    
    # parser.add_argument('--ckpt', type=str, default='/mnt/hdd/pointpillar_A9_dataset_checkpoints/output/tools/cfgs/A9_models/pointpillar_new/default/ckpt/checkpoint_epoch_200.pth', help='checkpoint to start from')
    # parser.add_argument('--ckpt', type=str, default='/mnt/8TB/PJW/BG-paper_weight/PointPillars/A9/BG_pointpillars_best_weight_a9.pth', help='specify the pretrained model')
    # parser.add_argument('--ckpt', type=str, default='/mnt/EFFB-D182/PJW/pointpillars_best/checkpoint_epoch_175.pth', help='specify the pretrained model')

    # parser.add_argument('--ckpt', type=str, default='/mnt/8TB/PJW/pointpillar_A9_dataset_checkpoints/output/home/ivpg/PJW/CUDA-PointPillars/OpenPCDet/tools/cfgs/A9_models/pointpillar_new_1_v2_dual_pointnet_background_0.005_8_2nd/bs-8-lr-0.001/ckpt/checkpoint_epoch_180.pth', help='specify the pretrained model')
    parser.add_argument('--ckpt', type=str, default='/mnt/8TB/PJW/pointpillar_A9_dataset_checkpoints/output/tools/cfgs/A9_models/pointpillar_new_1_v2_best_2nd/bs-8-lr-0.001-3rd/ckpt/checkpoint_epoch_155.pth', help='specify the pretrained model')

    # parser.add_argument('--ckpt', type=str, default='/media/ivpg/8TB/PJW/pointpillar_a9/output/tools/cfgs/withrobot_models/pointpillar_label_in_camera_fov/default/ckpt/checkpoint_epoch_200.pth', help='checkpoint to start from')
    # parser.add_argument('--ckpt', type=str, default='/mnt/hdd/pointpillar_A9_dataset_checkpoints/output/tools/cfgs/A9_models/pointpillar_new_1_v2_dual_pointnet_background/default/revise_version/checkpoint_epoch_175.pth', help='checkpoint to start from')
    # parser.add_argument('--ckpt', type=str, default='/mnt/hdd/pointpillar_A9_dataset_checkpoints/output/tools/cfgs/RC_models/pointpillar/default/ckpt/checkpoint_epoch_200.pth', help='checkpoint to start from') original pointpillar weight here
    # parser.add_argument('--ckpt', type=str, default='/mnt/hdd/pointpillar_A9_dataset_checkpoints/output/tools/cfgs/A9_models/pointpillar_old_no_aug/default/ckpt/checkpoint_epoch_200.pth', help='checkpoint to start from')
    
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default="/mnt/8TB/PJW/pointpillar_A9_dataset_checkpoints/output/tools/cfgs/A9_models/pointpillar_new_1_v2_best_2nd/bs-8-lr-0.001-3rd/ckpt", help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, test_set, args, eval_output_dir, logger, epoch_id, dist_test=False):
    
    print(args.ckpt)
    # load checkpoint
    # model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    
    ckpt_ = torch.load(args.ckpt,map_location="cuda:0")
    model = ckpt_['model'].float().cuda()
    print(model)
    print("model is successfully loaded!!")
    model = pruned_Model(model,model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    
    model.cuda()
    
    # for a in model.vfe.pfn_layers:
    #         print(a.linear.weight)
    #         print(a.norm.bias)
                
    dummy_voxels = torch.zeros(
        (MAX_VOXELS, MAX_NUM_POINTS, 4),
        dtype=torch.float32,
        device='cuda:0')

    dummy_voxel_idxs = torch.zeros(
        (MAX_VOXELS, 4),
        dtype=torch.int32,
        device='cuda:0')

    dummy_voxel_num = torch.zeros(
        (1,),
        dtype=torch.int32,
        device='cuda:0')

    dummy_input = dict()
    dummy_input['voxels'] = dummy_voxels
    dummy_input['voxel_num_points'] = dummy_voxel_num
    dummy_input['voxel_coords'] = dummy_voxel_idxs
    # dummy_input['batch_size'] = 1
    
    dummy_input_ = dict()
    dummy_input_['voxels'] = dummy_voxels
    dummy_input_['voxel_num_points'] = dummy_voxel_num
    dummy_input_['voxel_coords'] = dummy_voxel_idxs
    dummy_input_['batch_size'] = 1
    # base_macs_, base_nparams_ = tp.utils.count_ops_and_params(model, dummy_input_)
    # print("Before Pruning: MACs=%f G, #Params=%f G"%(base_macs_/1e9, base_nparams_/1e9))
    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file
    )


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(model, test_loader, test_set, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    total_time = 0
    first_eval = True

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        ckpt_ = torch.load(cur_ckpt,map_location="cuda:0")
        model = ckpt_['model'].float().cuda()
        model = pruned_Model(model,model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    
        model.cuda()
        
        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.eval_one_epoch(
            cfg, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir, save_to_file=args.save_to_file
        )

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.OUTPUT_DIR / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    with torch.no_grad():
        if args.eval_all:
            repeat_eval_ckpt(model, test_loader, test_set, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
        else:
            eval_single_ckpt(model, test_loader, test_set, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)


if __name__ == '__main__':
    main()
