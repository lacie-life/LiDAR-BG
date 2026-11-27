import pickle
import time

import numpy as np
import torch
import tqdm
import os

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False
    

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    
    start_time = time.time()
    mean_h = 0
    mean_w = 0
    mean_l = 0
    cnt = 0

    total_infer_time = 0
    total_frames = 0
    torch.cuda.reset_peak_memory_stats()
    
    for i, batch_dict in enumerate(dataloader):
        
        load_data_to_gpu(batch_dict)
        
        # print(batch_dict['voxels'].shape)s
        with torch.no_grad():
            torch.cuda.synchronize() 
            infer_start = time.time()

            pred_dicts, ret_dict = model(batch_dict)

            torch.cuda.synchronize()
            total_infer_time += (time.time() - infer_start)
            total_frames += batch_dict['batch_size']

        disp_dict = {}
        
        statistics_info(cfg, ret_dict, metric, disp_dict)

        ##### for complex YOLO AUC AP #######
        pred_box = pred_dicts[0]['pred_boxes'].cpu().numpy() # N,7                
        pred_conf_score = pred_dicts[0]['pred_scores'].cpu().numpy()[:,np.newaxis] # N,1
        pred_cls = pred_dicts[0]['pred_labels'].cpu().numpy()[:,np.newaxis].astype(np.float32) # N,1

        temp = pred_box[:,1].copy()
        pred_box[:,1] = pred_box[:,0]
        pred_box[:,0] = temp # change x & y position
        
        temp = pred_box[:,3].copy()
        pred_box[:,3] = pred_box[:,5]
        pred_box[:,5] = temp # change l & h position
        
        im = np.sin(-pred_box[:,6])[:,np.newaxis]
        re = np.cos(-pred_box[:,6])[:,np.newaxis]
        
        comp_output = np.zeros([pred_box.shape[0],11],dtype = np.float32)
        comp_output[:,:6] = pred_box[:,:6]
        comp_output[:,6:7] = im[:,:]
        comp_output[:,7:8] = re[:,:]
        comp_output[:,8:9] = pred_conf_score[:,:]
        comp_output[:,10:] = pred_cls[:,:]
        # BG_Pointpillars_best_weight_A9_180
        np.save(os.path.join("/home/ivpg/PJW/CUDA-PointPillars/OpenPCDet/Pointpillars_best_weight_A9_155_3rd",batch_dict['frame_id'][0]+"_pointpillar_a9.npy"),comp_output)
        
        ##### for complex YOLO AUC AP #######

        ### from complex YOLO results ###
        # pred_dicts = []
        # temp = {}
        # temp['pred_boxes'] = torch.tensor(np.load(os.path.join("/home/ivpg/PJW/CUDA-PointPillars/OpenPCDet/complex_yolov7_3D_a9_v2_results_openpcdet_forground",batch_dict['frame_id'][0]+"_pred_boxes.npy"))).cuda()
        # temp['pred_scores'] = torch.tensor(np.load(os.path.join("/home/ivpg/PJW/CUDA-PointPillars/OpenPCDet/complex_yolov7_3D_a9_v2_results_openpcdet_forground",batch_dict['frame_id'][0]+"_pred_scores.npy"))).cuda()
        # temp['pred_labels'] = torch.tensor(np.load(os.path.join("/home/ivpg/PJW/CUDA-PointPillars/OpenPCDet/complex_yolov7_3D_a9_v2_results_openpcdet_forground",batch_dict['frame_id'][0]+"_pred_labels.npy")).astype(np.int32)).cuda()
        
        # print(batch_dict['gt_boxes'])
        # try:
        #     temp['pred_boxes'] = torch.tensor(np.load(os.path.join("/home/jiwon/3d_object_detection/Complex-YOLO3D_kitti/src/complex_yolov7_3D_kitti_results_without_margin",batch_dict['frame_id'][0]+"_pred_boxes.npy"))).cuda()
        # except:
        #     temp['pred_boxes'] = torch.tensor(np.array([])).cuda()
        # try: 
        #     temp['pred_scores'] = torch.tensor(np.load(os.path.join("/home/jiwon/3d_object_detection/Complex-YOLO3D_kitti/src/complex_yolov7_3D_kitti_results_without_margin",batch_dict['frame_id'][0]+"_pred_scores.npy"))).cuda()
        # except:
        #     temp['pred_scores'] = torch.tensor(np.array([])).cuda()      
        # try:      
        #     temp['pred_labels'] = torch.tensor(np.load(os.path.join("/home/jiwon/3d_object_detection/Complex-YOLO3D_kitti/src/complex_yolov7_3D_kitti_results_without_margin",batch_dict['frame_id'][0]+"_pred_labels.npy")).astype(np.int32)).cuda()
        # except:
        #     temp['pred_labels'] = torch.tensor(np.array([])).cuda()  
        
        # pred_dicts.append(temp)
        
        ### from complex YOLO results ###
        
        # print(pred_dicts)
        
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        
        det_annos += annos
        
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()
            
        # pred_box = pred_dicts[0]['pred_boxes'].cpu().numpy() # N,7                
        # pred_conf_score = pred_dicts[0]['pred_scores'].cpu().numpy()[:,np.newaxis] # N,1
        # pred_cls = pred_dicts[0]['pred_labels'].cpu().numpy()[:,np.newaxis].astype(np.float32) # N,1
        
        # temp = pred_box[:,1].copy()
        # pred_box[:,1] = pred_box[:,0]
        # pred_box[:,0] = temp # change x & y position

        try:
            a = np.squeeze(pred_cls == 3)
            cnt += np.sum(a)
            # print(cnt)
            mean_l += np.sum(pred_box[a,3])
            mean_w += np.sum(pred_box[a,4])
            mean_h += np.sum(pred_box[a,5])
        except:
            continue
        
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print(mean_l/cnt)
    # print(mean_w/cnt)
    # print(mean_h/cnt)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if cfg.LOCAL_RANK == 0:
        avg_fps = total_frames / total_infer_time
        avg_latency = (total_infer_time / total_frames) * 1000
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) 

        print('\n' + '='*50)
        print(f'REAL-WORLD DEPLOYMENT METRICS')
        print(f'Inference Speed: {avg_fps:.2f} FPS')
        print(f'Latency:         {avg_latency:.2f} ms/frame')
        print(f'Peak VRAM Usage: {peak_vram:.2f} MiB')
        print('='*50 + '\n')

        logger.info(f'Wall-clock FPS: {avg_fps:.2f}')
        logger.info(f'Average Latency: {avg_latency:.2f} ms/frame')
        logger.info(f'Peak VRAM Usage: {peak_vram:.2f} MiB')
        
        ret_dict['performance/fps'] = avg_fps
        ret_dict['performance/vram_mib'] = peak_vram

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict

def eval_one_epoch_dsp(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    
    dsp_results_path = "/media/ivpg/8TB/PJW/snpe_228/pointpillar_6490_a9_v2/pointpillar_RGB_A9_dataset"
    cls_head = "252.raw"
    box_head = "253.raw"
    dir_cls_head = "256.raw"   
    
    start_time = time.time()
    
    for i, batch_dict in enumerate(dataloader):
        
        load_data_to_gpu(batch_dict)
        
        cls_preds = torch.tensor(np.fromfile(os.path.join(os.path.join(dsp_results_path,"Result_%d" %(i)),cls_head),dtype = np.float32).reshape(1,304,304,18)).to("cuda:0").contiguous()
        box_preds = torch.tensor(np.fromfile(os.path.join(os.path.join(dsp_results_path,"Result_%d" %(i)),box_head),dtype = np.float32).reshape(1,304,304,42)).to("cuda:0").contiguous()
        dir_cls_preds = torch.tensor(np.fromfile(os.path.join(os.path.join(dsp_results_path,"Result_%d" %(i)),dir_cls_head),dtype = np.float32).reshape(1,304,304,12)).to("cuda:0").contiguous()
        
        batch_dict['cls_preds'] = cls_preds
        batch_dict['box_preds'] = box_preds      
        batch_dict['dir_cls_preds'] = dir_cls_preds 
               
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
