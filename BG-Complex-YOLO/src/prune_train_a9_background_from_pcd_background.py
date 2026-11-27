import time
import numpy as np
import sys
import random
import os
import warnings
import cv2
import math
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from tqdm import tqdm
from models.experimental import attempt_load_train
sys.path.append('./')


from data_process.a9_dataloader_background_from_pcd_background import create_train_dataloader, create_val_dataloader
from models.model_utils import create_model, make_data_parallel, get_num_parameters
from utils.train_utils import create_optimizer, create_lr_scheduler, get_saved_state, save_checkpoint
from utils.train_utils import reduce_tensor, to_python_float, get_tensorboard_log
from utils.misc import AverageMeter, ProgressMeter
from utils.logger import Logger
from config.train_config import parse_train_configs
from evaluate_a9 import evaluate_mAP_a9

from data_process import a9_bev_utils                                                                                                                                                                   
import config.a9_config as cnf
from utils.visualization_utils import merge_rgb_to_bev



### yolov7 ###
from models.yolo import Model
from torchsummaryX import summary                                                                                                   
from torch.cuda import amp
# f = open("./a9_train_list_kitti_range_100%.txt","w")

### for pruning ###
import torch_pruning as tp
from functools import partial
configs = parse_train_configs()

def progressive_pruning(pruner, model, speed_up, example_inputs):
        model.eval()
        base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_speed_up = 1
        while current_speed_up < configs.speed_up:
            pruner.step(interactive=False)
            pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
            current_speed_up = float(base_ops) / pruned_ops
            if pruner.current_step == pruner.iterative_steps:
                break
            print('current_speed_up %f'%current_speed_up)
        return current_speed_up

def get_pruner(model, example_inputs):
        configs.sparsity_learning = False
        if configs.method == "random":
            imp = tp.importance.RandomImportance()
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=configs.global_pruning)
        elif configs.method == "l1":
            imp = tp.importance.MagnitudeImportance(p=1)
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=configs.global_pruning)
        elif configs.method == "lamp":
            imp = tp.importance.LAMPImportance(p=2)
            pruner_entry = partial(tp.pruner.BNScalePruner, global_pruning=configs.global_pruning)
        elif configs.method == "slim":
            configs.sparsity_learning = True
            imp = tp.importance.BNScaleImportance()
            pruner_entry = partial(tp.pruner.BNScalePruner, reg=configs.reg, global_pruning=configs.global_pruning)
        elif configs.method == "group_slim":
            configs.sparsity_learning = True
            imp = tp.importance.BNScaleImportance()
            pruner_entry = partial(tp.pruner.BNScalePruner, reg=configs.reg, global_pruning=configs.global_pruning, group_lasso=True)
        elif configs.method == "group_norm":
            imp = tp.importance.GroupNormImportance(p=2)
            pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=configs.global_pruning)
        elif configs.method == "group_sl":
            configs.sparsity_learning = True
            imp = tp.importance.GroupNormImportance(p=2, normalizer='max') # normalized by the maximum score for CIFAR
            pruner_entry = partial(tp.pruner.GroupNormPruner, reg=configs.reg, global_pruning=configs.global_pruning)
        elif configs.method == "growing_reg":
            configs.sparsity_learning = True
            imp = tp.importance.GroupNormImportance(p=2)
            pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=configs.reg, delta_reg=configs.delta_reg, global_pruning=configs.global_pruning)
        else:
            raise NotImplementedError
        
        #opt.is_accum_importance = is_accum_importance
        pruning_ratio_dict = {}

        # ignore output layers
        ignored_layers = []
        from models.yolo import Detect, IDetect
        from models.common import ImplicitA, ImplicitM
        from models.yolo_layer_v7 import YoloLayer_v7
        for m in model.modules():
            if isinstance(m, (Detect,IDetect,YoloLayer_v7)):
                ignored_layers.append(m.m)

        unwrapped_parameters = []
        for m in model.modules():
            if isinstance(m, (ImplicitA,ImplicitM)):
                unwrapped_parameters.append((m.implicit,1)) # pruning 1st dimension of implicit matrix

        # Here we fix iterative_steps=200 to prune the model progressively with small steps 
        # until the required speed up is achieved.
        pruner = pruner_entry(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=configs.iterative_steps,
            pruning_ratio=1.0,
            pruning_ratio_dict=pruning_ratio_dict,
            max_pruning_ratio=configs.max_pruning_ratio,
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters,
        )
        return pruner
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
def main():
    
    # Re-produce results
    if configs.seed is not None:
        random.seed(configs.seed)
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if configs.gpu_idx is not None:
        print('You have chosen a specific GPU. This will completely disable data parallelism.')

    if configs.dist_url == "env://" and configs.world_size == -1:
        configs.world_size = int(os.environ["WORLD_SIZE"])

    configs.distributed = configs.world_size > 1 or configs.multiprocessing_distributed

    if configs.multiprocessing_distributed:
        configs.world_size = configs.ngpus_per_node * configs.world_size
        mp.spawn(main_worker, nprocs=configs.ngpus_per_node, args=(configs,))
    else:
        main_worker(configs.gpu_idx, configs)


def main_worker(gpu_idx, configs):
    configs.gpu_idx = gpu_idx
    configs.device = torch.device('cpu' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx))

    if configs.distributed:
        if configs.dist_url == "env://" and configs.rank == -1:
            configs.rank = int(os.environ["RANK"])
        if configs.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            configs.rank = configs.rank * configs.ngpus_per_node + gpu_idx

        dist.init_process_group(backend=configs.dist_backend, init_method=configs.dist_url,
                                world_size=configs.world_size, rank=configs.rank)
        configs.subdivisions = int(64 / configs.batch_size / configs.ngpus_per_node)
    else:
        configs.subdivisions = int(64 / configs.batch_size)

    configs.is_master_node = (not configs.distributed) or (
            configs.distributed and (configs.rank % configs.ngpus_per_node == 0))

    if configs.is_master_node:
        logger = Logger(configs.logs_dir, configs.saved_fn)
        logger.info('>>> Created a new logger')
        logger.info('>>> configs: {}'.format(configs))
        tb_writer = SummaryWriter(log_dir=os.path.join(configs.logs_dir, 'tensorboard'))
    else:
        logger = None
        tb_writer = None
    
    logger = None
    tb_writer = None
    
    # model
    # model = create_model(configs).to(configs.device)
    
    # hyp =  yaml.load(configs.yolov7_hyp,Loader =yaml.SafeLoader)
        
    model = Model(configs.yolov7_cfg, ch=6, nc=len(cnf.class_list), anchors=None).to(configs.device)  # create
    summary(model,torch.rand(1,6,608,608).to(configs.device))
    
    # model = combine_yolov7_head(model)
    model.eval()
    print(model)
    example_inputs = torch.randn(1, 6, 608, 608).to(configs.device)
    
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    
    pruner = get_pruner(model, example_inputs=example_inputs)
    
    progressive_pruning(pruner, model, speed_up=configs.speed_up, example_inputs=example_inputs)
    
    pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(model)
    print("Before Pruning: MACs=%f G, #Params=%f G"%(base_macs/1e9, base_nparams/1e9))
    print("After Pruning: MACs=%f G, #Params=%f G"%(pruned_macs/1e9, pruned_nparams/1e9))
    #########    
    # load weight from a checkpoint
    if configs.pretrained_path is not None:
        assert os.path.isfile(configs.pretrained_path), "=> no checkpoint found at '{}'".format(configs.pretrained_path)
        model.load_state_dict(torch.load(configs.pretrained_path))
        if logger is not None:
            logger.info('loaded pretrained model at {}'.format(configs.pretrained_path))

    if configs.resume_path is not None:
        assert os.path.isfile(configs.resume_path), "=> no checkpoint found at '{}'".format(configs.resume_path)
        model = attempt_load_train(configs.resume_path, map_location=configs.device)  # load FP32 model
        if logger is not None:
            logger.info('resume training model from checkpoint {}'.format(configs.resume_path))


    # Data Parallel
    # model = make_data_parallel(model, configs)

    # Make sure to create optimizer after moving the model to cuda
    optimizer = create_optimizer(configs, model)
    lr_scheduler = create_lr_scheduler(optimizer, configs)
    configs.step_lr_in_epoch = True if configs.lr_type in ['multi_step'] else False

    # resume optimizer, lr_scheduler from a checkpoint
    if configs.resume_path is not None:
        utils_path = configs.resume_path.replace('Model_', 'Utils_')
        assert os.path.isfile(utils_path), "=> no checkpoint found at '{}'".format(utils_path)
        utils_state_dict = torch.load(utils_path, map_location='cuda:{}'.format(configs.gpu_idx))
        optimizer.load_state_dict(utils_state_dict['optimizer'])
        lr_scheduler.load_state_dict(utils_state_dict['lr_scheduler'])
        configs.start_epoch = utils_state_dict['epoch'] + 1

    if configs.is_master_node:
        num_parameters = get_num_parameters(model)
        # logger.info('number of trained parameters of the model: {}'.format(num_parameters))

    if logger is not None:
        logger.info(">>> Loading dataset & getting dataloader...")
    # Create dataloader
    train_dataloader, train_sampler = create_train_dataloader(configs)
    if logger is not None:
        logger.info('number of batches in training set: {}'.format(len(train_dataloader)))

    if configs.evaluate:
        val_dataloader = create_val_dataloader(configs)
        precision, recall, AP, f1, ap_class = evaluate_mAP_a9(val_dataloader, model, configs, None)
        print('Evaluate - precision: {}, recall: {}, AP: {}, f1: {}, ap_class: {}'.format(precision, recall, AP, f1,
                                                                                          ap_class))
        print('mAP {}'.format(AP.mean()))
        return

    hightest_val_ap = 0.0

    for epoch in range(configs.start_epoch, configs.num_epochs + 1):
        if logger is not None:
            logger.info('{}'.format('*-' * 40))
            logger.info('{} {}/{} {}'.format('=' * 35, epoch, configs.num_epochs, '=' * 35))
            logger.info('{}'.format('*-' * 40))
            logger.info('>>> Epoch: [{}/{}]'.format(epoch, configs.num_epochs))

        if configs.distributed:
            train_sampler.set_epoch(epoch)
        
        # train for one epoch
        train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, logger, tb_writer)
       
        # model_state_dict, utils_state_dict = get_saved_state(model, optimizer, lr_scheduler, epoch, configs)
        # save_checkpoint(configs.checkpoints_dir, 'best', model_state_dict, utils_state_dict, epoch)
        
        if not configs.no_val:
            val_dataloader = create_val_dataloader(configs)
            print('number of batches in val_dataloader: {}'.format(len(val_dataloader)))
            precision, recall, AP, f1, ap_class = evaluate_mAP_a9(val_dataloader, model, configs, logger)
            print('AP : {} '.format(AP.mean()))
            val_metrics_dict = {
                'precision': precision.mean(),
                'recall': recall.mean(),
                'AP': AP.mean(),
                'f1': f1.mean(),
                'ap_class': ap_class.mean()
            }
            # if ((epoch % configs.checkpoint_freq) == 0):
            #     model_state_dict, utils_state_dict = get_saved_state(model, optimizer, lr_scheduler, epoch, configs)
            # #     save_checkpoint(configs.checkpoints_dir, 'best', model_state_dict, utils_state_dict, epoch)
            # if AP.mean() > hightest_val_ap:
            #     hightest_val_ap = AP.mean()
            #     model_state_dict, utils_state_dict = get_saved_state(model, optimizer, lr_scheduler, epoch, configs)
            #     save_checkpoint(configs.checkpoints_dir, 'best', model_state_dict, utils_state_dict, epoch)

            if tb_writer is not None:
                tb_writer.add_scalars('Validation', val_metrics_dict, epoch)

        # Save checkpoint
        if configs.is_master_node and ((epoch % configs.checkpoint_freq) == 0):
            model_state_dict, utils_state_dict = get_saved_state(model, optimizer, lr_scheduler, epoch, configs)
            save_checkpoint(configs.checkpoints_dir, configs.saved_fn, model_state_dict, utils_state_dict, epoch)

        if not configs.step_lr_in_epoch:
            lr_scheduler.step()
            if tb_writer is not None:
                tb_writer.add_scalar('LR', lr_scheduler.get_lr()[0], epoch)

    if tb_writer is not None:
        tb_writer.close()
    if configs.distributed:
        cleanup()


def cleanup():
    dist.destroy_process_group()


def train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, logger, tb_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(len(train_dataloader), [batch_time, data_time, losses],
                             prefix="Train - Epoch: [{}/{}]".format(epoch, configs.num_epochs))

    num_iters_per_epoch = len(train_dataloader)

    # switch to train mode
    model.train()
    # model.eval()
    
    # scaler = amp.GradScaler(enabled=True)
    start_time = time.time()
    
    for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
        data_time.update(time.time() - start_time)
        img_path, imgs, targets = batch_data
        
        # img_raw = np.array(imgs.permute(0,2,3,1).squeeze(dim = 0),dtype = np.float32)
        # cv2.imshow("ss",img_raw)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # img_raw.tofile(os.path.join('./a9_train_raw_kitti_range',img_path[0].split('/')[-1][:-4]+".raw"))
        # f.write(os.path.join("/data/local/tmp/pjw/a9_train_raw_kitti_range/",img_path[0].split('/')[-1][:-4]+".raw"))
        # f.write('\n')       
             
        global_step = num_iters_per_epoch * (epoch - 1) + batch_idx + 1

        batch_size = imgs.size(0)

        targets = targets.to(configs.device, non_blocking=True)
        imgs = imgs.to(configs.device, non_blocking=True)
        total_loss, outputs = model(imgs,targets)
        
        # print(total_loss)
        # print(outputs[0].shape,outputs[1].shape,outputs[2].shape)
        
        # total_loss = compute_loss(outputs,targets)
        #total_loss = compute_loss_yolov4(outputs,targets)
        
        # For torch.nn.DataParallel case
        if (not configs.distributed) and (configs.gpu_idx is None):
            total_loss = torch.mean(total_loss)

        # compute gradient and perform backpropagation
        total_loss.backward()
        
        # total_loss.backward()
        if global_step % configs.subdivisions == 0:
            optimizer.step()
            # Adjust learning rate
            if configs.step_lr_in_epoch:
                lr_scheduler.step()
                if tb_writer is not None:
                    tb_writer.add_scalar('LR', lr_scheduler.get_lr()[0], global_step)
            # zero the parameter gradients
            optimizer.zero_grad()

        if configs.distributed:
            reduced_loss = reduce_tensor(total_loss.data, configs.world_size)
        else:
            reduced_loss = total_loss.data
        losses.update(to_python_float(reduced_loss), batch_size)
        # measure elapsed time
        # torch.cuda.synchronize()
        batch_time.update(time.time() - start_time)

        # if tb_writer is not None:
        #     if (global_step % configs.tensorboard_freq) == 0:
        #         tensorboard_log = get_tensorboard_log(model)
        #         tb_writer.add_scalar('avg_loss', losses.avg, global_step)
        #         for layer_name, layer_dict in tensorboard_log.items():
        #             tb_writer.add_scalars(layer_name, layer_dict, global_step)

        # Log message
        if logger is not None:
            if (global_step % configs.print_freq) == 0:
                logger.info(progress.get_message(batch_idx))

        start_time = time.time()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        try:
            cleanup()
            sys.exit(0)
        except SystemExit:
            os._exit(0)
