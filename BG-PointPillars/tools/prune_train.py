import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_2, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from pcdet.models.detectors.detector3d_template import Detector3DTemplate
from pcdet.models.dense_heads.anchor_head_single import AnchorHeadSingle,AnchorHeadSingle_test
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model

import torch_pruning as tp
from functools import partial

class pruned_Model(nn.Module):
    def __init__(self, original_model):
        super(pruned_Model, self).__init__()
        self.vfe = original_model.vfe
        self.map_to_bev_module = original_model.map_to_bev_module
        self.backbone_2d = original_model.backbone_2d
        # self.dense_head = original_model.dense_head
        
    def forward(self, x):
        x = self.vfe(x)
        x = self.map_to_bev_module(x)
        x = self.backbone_2d(x)
        # x = self.dense_head(x)
        
        return x

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
            # point_cloud_range=np.array([0, -80.0, -8.0, 160.0, 80.0, -1.0]),
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

# class pruned_Model_full(Detector3DTemplate):
#     def __init__(self, original_model,input_channels,model_cfg, num_class, dataset):
#         super(pruned_Model_full, self).__init__(model_cfg, num_class, dataset)
#         self.vfe = original_model.vfe
#         self.map_to_bev_module = original_model.map_to_bev_module
#         self.backbone_2d = original_model.backbone_2d
#         self.dense_head = AnchorHeadSingle_test(model_cfg=self.model_cfg.DENSE_HEAD,
#             input_channels=input_channels,
#             num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
#             class_names=self.class_names,
#             grid_size=np.array([608,608,1]),
#             point_cloud_range=np.array([0.0, -80.0, -8.0, 160.0, 80.0, -1.0]),
#             predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
#             voxel_size=[0.26315, 0.26315, 7.0]
#         )

#     def forward(self, x):
#         x = self.vfe(x)
#         x = self.map_to_bev_module(x)
#         x = self.backbone_2d(x)  
#         x = self.dense_head(x)
        
#         if self.training:
#             loss, tb_dict, disp_dict = self.get_training_loss()

#             ret_dict = {
#                 'loss': loss
#             }
#             return ret_dict, tb_dict, disp_dict
#         else:
#             pred_dicts, recall_dicts = self.post_processing(x)
#             return pred_dicts, recall_dicts
        
#     def get_training_loss(self):
#         disp_dict = {}

#         loss_rpn, tb_dict = self.dense_head.get_loss()
#         tb_dict = {
#             'loss_rpn': loss_rpn.item(),
#             **tb_dict
#         }

#         loss = loss_rpn
#         return loss, tb_dict, disp_dict
    
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/withrobot_models/pointpillar_label_in_camera_fov.yaml', help='specify the config for training')
    # parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/withrobot_models/DHIAHpillars.yaml', help='specify the config for training')
    parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/A9_models/pointpillar_new_1_v2_best_2nd.yaml', help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=8, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=200, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=5, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--gpu_idx', default="0", type=str,help='GPU index to use.')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=1, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    ###################### for pruning #############################
    
    parser.add_argument("--method", type=str, default='random')
    parser.add_argument("--speed-up", type=float, default=1.0)
    parser.add_argument("--max-pruning-ratio", type=float, default=1.0)
    parser.add_argument("--soft-keeping-ratio", type=float, default=0.0)
    parser.add_argument("--reg", type=float, default=5e-4)
    parser.add_argument("--delta_reg", type=float, default=1e-4, help='for growing regularization')
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    parser.add_argument("--global-pruning", action="store_true", default=True)
    parser.add_argument("--sl-total-epochs", type=int, default=100, help="epochs for sparsity learning")
    parser.add_argument("--sl-lr", default=0.01, type=float, help="learning rate for sparsity learning")
    parser.add_argument("--sl-lr-decay-milestones", default="60,80", type=str, help="milestones for sparsity learning")
    parser.add_argument("--sl-reg-warmup", type=int, default=0, help="epochs for sparsity learning")
    parser.add_argument("--sl-restore", type=str, default=None)
    parser.add_argument("--iterative-steps", default=400, type=int)
    
    ################################################################
    
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    cfg.output_dir = Path()
    
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
        
    def progressive_pruning(pruner, model, speed_up, example_inputs):
            model.eval()
            
            base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
            current_speed_up = 1
            while current_speed_up < args.speed_up:
                pruner.step(interactive=False)
                pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
                current_speed_up = float(base_ops) / pruned_ops
                if pruner.current_step == pruner.iterative_steps:
                    break
                print('current_speed_up %f'%current_speed_up)
            return current_speed_up

    def get_pruner(model, example_inputs):
            args.sparsity_learning = False
            if args.method == "random":
                imp = tp.importance.RandomImportance()
                pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
            elif args.method == "l1":
                imp = tp.importance.MagnitudeImportance(p=1)
                pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
            elif args.method == "lamp":
                imp = tp.importance.LAMPImportance(p=2)
                pruner_entry = partial(tp.pruner.BNScalePruner, global_pruning=args.global_pruning)
            elif args.method == "slim":
                args.sparsity_learning = True
                imp = tp.importance.BNScaleImportance()
                pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning)
            elif args.method == "group_slim":
                args.sparsity_learning = True
                imp = tp.importance.BNScaleImportance()
                pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning, group_lasso=True)
            elif args.method == "group_norm":
                imp = tp.importance.GroupNormImportance(p=2)
                pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=args.global_pruning)
            elif args.method == "group_sl":
                args.sparsity_learning = True
                imp = tp.importance.GroupNormImportance(p=2, normalizer='max') # normalized by the maximum score for CIFAR
                pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
            elif args.method == "growing_reg":
                args.sparsity_learning = True
                imp = tp.importance.GroupNormImportance(p=2)
                pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=args.reg, delta_reg=args.delta_reg, global_pruning=args.global_pruning)
            else:
                raise NotImplementedError
            
            #opt.is_accum_importance = is_accum_importance
            pruning_ratio_dict = {}
            # for m in model.modules():
            #     print(m)
            # ignore output layers
            ignored_layers = []

            from pcdet.models.dense_heads.anchor_head_single import AnchorHeadSingle
            from pcdet.models.backbones_2d.map_to_bev.pointpillar_scatter import PointPillarScatter
            # for m in model.modules():
            #     print(m)
            for i, m in enumerate(model.modules()):
                # print(i,m)
                
                if isinstance(m, (AnchorHeadSingle)):
                    ignored_layers.append(m.conv_cls)
                    ignored_layers.append(m.conv_box)
                    ignored_layers.append(m.conv_dir_cls)
                
                    
                # if isinstance(m, (PointPillarScatter)):
                    # ignored_layers.append(m)
                    
            unwrapped_parameters = []
            # for m in model.modules():
            #     if isinstance(m, (ImplicitA,ImplicitM)):
            #         unwrapped_parameters.append((m.implicit,1)) # pruning 1st dimension of implicit matrix

            # Here we fix iterative_steps=200 to prune the model progressively with small steps 
            # until the required speed up is achieved.
            
            pruner = pruner_entry(
                model,
                example_inputs,
                importance=imp,
                iterative_steps=args.iterative_steps,
                pruning_ratio=1.0,
                pruning_ratio_dict=pruning_ratio_dict,
                max_pruning_ratio=args.max_pruning_ratio,
                ignored_layers=ignored_layers,
                unwrapped_parameters=unwrapped_parameters,
            )
            return pruner

    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)
    
    
    output_dir = cfg.OUTPUT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    # output_dir = cfg.OUTPUT_DIR
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    # gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES']
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model.eval()

    MAX_VOXELS = 32000
    MAX_NUM_POINTS = 32
    # dummy_voxels = torch.zeros(
    #     (MAX_VOXELS, MAX_NUM_POINTS, 4),
    #     dtype=torch.float32,
    #     device='cuda:0')

    # dummy_voxel_idxs = torch.zeros(
    #     (MAX_VOXELS, 4),
    #     dtype=torch.int32,
    #     device='cuda:0')

    # dummy_voxel_num = torch.zeros(
    #     (1,),
    #     dtype=torch.int32,
    #     device='cuda:0')

    # dummy_input = dict()
    # dummy_input['voxels'] = dummy_voxels
    # dummy_input['voxel_num_points'] = dummy_voxel_num
    # dummy_input['voxel_coords'] = dummy_voxel_idxs
    # # dummy_input['batch_size'] = 1
    
    # dummy_input_ = dict()
    # dummy_input_['voxels'] = dummy_voxels
    # dummy_input_['voxel_num_points'] = dummy_voxel_num
    # dummy_input_['voxel_coords'] = dummy_voxel_idxs
    # dummy_input_['batch_size'] = 1
    
    # base_macs_, base_nparams_ = tp.utils.count_ops_and_params(model, dummy_input_)
    
    # model = pruned_Model(model)

    # base_macs, base_nparams = tp.utils.count_ops_and_params(model, dummy_input)
    # # print("Before Pruning: MACs=%f G, #Params=%f G"%(base_macs/1e9, base_nparams/1e9))
    
    # pruner = get_pruner(model, example_inputs=dummy_input)
    
    # progressive_pruning(pruner, model, speed_up=args.speed_up, example_inputs=dummy_input)
    
    # # print(model)
    # input_channels = model(dummy_input)
    # input_channels = input_channels['spatial_features_2d'].shape[1]
    
    # model = pruned_Model_full(model,input_channels,model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    
    # pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model, dummy_input_)
    
    # print("Before Pruning: MACs=%f G, #Params=%f G"%(base_macs_/1e9, base_nparams_/1e9))
    # print("After Pruning: MACs=%f G, #Params=%f G"%(pruned_macs/1e9, pruned_nparams/1e9))
    
    

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    # for a in model.vfe.pfn_layers:
    
    #     for param in a.linear.parameters():
    #         print(param)

        # for param in a.conv1x1.parameters():
        #     print(param)    

        # for param in a.norm.parameters():
        #     param.requires_grad = False
    
    if args.pretrained_model is not None:
        # model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)
        ckpt_ = torch.load(args.pretrained_model,map_location="cuda:0")
        model = ckpt_['model'].float()

        # for a in model.vfe.pfn_layers:
        
        #     for param in a.linear.parameters():
        #         print(param)

        # for a in model.vfe.pfn_layers:
        
        #     # for param in a.linear.parameters():
        #     #     param.requires_grad = False
        #     for param in a.conv1x1.parameters():
        #         param.requires_grad = False    

        #     for param in a.norm.parameters():
        #         param.requires_grad = False
    
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch
    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)  # Only evaluate the last args.num_epochs_to_eval epochs

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
