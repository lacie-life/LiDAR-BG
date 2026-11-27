import os
import numpy as np
import cv2
import random
from pypcd4 import PointCloud
from tools import *
import open3d as o3d

def create_voxel_grid(points, voxel_size):
    # points는 N,4 형태의 numpy 배열이라고 가정합니다.
    # 첫 번째 3개의 열은 좌표 (x, y, z), 네 번째 열은 추가 정보입니다.
    
    # 포인트 클라우드에서 좌표 부분을 분리합니다.
    coords = points[:, :3]
    
    # 각 좌표를 복셀 크기로 나누어 보셀 인덱스를 구합니다.
    voxel_indices = np.floor(coords / voxel_size).astype(np.int32)
    
    # 각 복셀 인덱스를 유일하게 만듭니다.
    unique_voxel_indices = np.unique(voxel_indices, axis=0)
    
    # 복셀 인덱스를 키로 하고, 각 복셀에 해당하는 포인트를 저장할 딕셔너리를 만듭니다.
    voxel_dict = {}
    for voxel_index in unique_voxel_indices:
        # 각 복셀 인덱스에 대해 해당하는 포인트를 찾습니다.
        mask = np.all(voxel_indices == voxel_index, axis=1)
        voxel_points = points[mask]
        
        # 복셀 인덱스를 튜플로 변환하여 딕셔너리의 키로 사용합니다.
        voxel_dict[tuple(voxel_index)] = voxel_points
    
    return voxel_dict

save_path = "./training/background_RC"
txt_file = open("./ImageSets/train.txt","r")

lines = txt_file.readlines()
random.shuffle(lines)

cnt_117 = 0
cnt_120 = 0
cnt_136 = 0
cnt_139 = 0

sample_num = 25
background_threshold_cnt = 20

voxel_size = ([0.1,0.1,0.1])

for line in lines:
    line = line.rstrip()
    pc_path = os.path.join("./training/pointcloud",line+".bin")

    if ("_117" in line) and (cnt_117 < sample_num):
        pc = np.fromfile(pc_path, dtype = np.float32).reshape(-1,4)
        pc_reduce = removePoints(pc, boundary)
        if cnt_117 == 0:
            pc_117_overlapped = pc_reduce  
        else:
            pc_117_overlapped = np.concatenate([pc_117_overlapped,pc_reduce],axis = 0)
            
        cnt_117 += 1
        
    elif ("_120" in line) and (cnt_120 < sample_num):
        pc = np.fromfile(pc_path, dtype = np.float32).reshape(-1,4)
        pc_reduce = removePoints(pc, boundary)
        
        if cnt_120 == 0:
            pc_120_overlapped = pc_reduce  
        else:
            pc_120_overlapped = np.concatenate([pc_120_overlapped,pc_reduce],axis = 0)
            
        cnt_120 += 1
        
    elif ("_136" in line)and (cnt_136 < sample_num):
        
        pc = np.fromfile(pc_path, dtype = np.float32).reshape(-1,4)
        pc_reduce = removePoints(pc, boundary)
        if cnt_136 == 0:
            pc_136_overlapped = pc_reduce  
        else:
            pc_136_overlapped = np.concatenate([pc_136_overlapped,pc_reduce],axis = 0)
            
        cnt_136 += 1
    
    elif ("_139" in line)and (cnt_139 < sample_num):
        
        pc = np.fromfile(pc_path, dtype = np.float32).reshape(-1,4)
        pc_reduce = removePoints(pc, boundary)
        if cnt_139 == 0:
            pc_139_overlapped = pc_reduce  
        else:
            pc_139_overlapped = np.concatenate([pc_139_overlapped,pc_reduce],axis = 0)
            
        cnt_139 += 1

voxel_grid_117 = create_voxel_grid(pc_117_overlapped,voxel_size)
print("117_voxel_created.\n")
voxel_grid_120 = create_voxel_grid(pc_120_overlapped,voxel_size)
print("120_voxel_created.\n")
voxel_grid_136 = create_voxel_grid(pc_136_overlapped,voxel_size)
print("136_voxel_created.\n")
voxel_grid_139 = create_voxel_grid(pc_136_overlapped,voxel_size)
print("139_voxel_created.\n")


background_117_list = []
for voxel_index, voxel_points in voxel_grid_117.items():
    
    if voxel_points.shape[0] >= background_threshold_cnt:
        
        coord_plus_intensity = np.mean(voxel_points,axis = 0)
        
        background_117_list.append(coord_plus_intensity)
    
background_117 = np.vstack(background_117_list)
print("background_117 Done.\n")


background_120_list = []
for voxel_index, voxel_points in voxel_grid_120.items():
    
    if voxel_points.shape[0] >= background_threshold_cnt:
        
        coord_plus_intensity = np.mean(voxel_points,axis = 0)
        
        background_120_list.append(coord_plus_intensity)
    
background_120 = np.vstack(background_120_list)
print("background_120 Done.\n")


background_136_list = []
for voxel_index, voxel_points in voxel_grid_136.items():
    
    if voxel_points.shape[0] >= background_threshold_cnt:
        
        coord_plus_intensity = np.mean(voxel_points,axis = 0)
        
        background_136_list.append(coord_plus_intensity)
    
background_136 = np.vstack(background_136_list)
print("background_136 Done.\n")


background_139_list = []
for voxel_index, voxel_points in voxel_grid_139.items():
    
    if voxel_points.shape[0] >= background_threshold_cnt:
        
        coord_plus_intensity = np.mean(voxel_points,axis = 0)
        
        background_139_list.append(coord_plus_intensity)
    
background_139 = np.vstack(background_139_list)
print("background_139 Done.\n")

background_117.tofile(os.path.join(save_path,"RC_117_background_by_3d_grid_25_15.bin"))
background_120.tofile(os.path.join(save_path,"RC_120_background_by_3d_grid_25_15.bin"))
background_136.tofile(os.path.join(save_path,"RC_136_background_by_3d_grid_25_15.bin"))
background_139.tofile(os.path.join(save_path,"RC_139_background_by_3d_grid_25_15.bin"))


# Open3D 포인트 클라우드 객체 생성
pcd_117 = o3d.geometry.PointCloud()
pcd_120 = o3d.geometry.PointCloud()
pcd_136 = o3d.geometry.PointCloud()
pcd_139 = o3d.geometry.PointCloud()


# numpy 배열의 x, y, z 좌표 부분을 사용하여 포인트 클라우드 설정
pcd_117.points = o3d.utility.Vector3dVector(background_117[:, :3])
pcd_120.points = o3d.utility.Vector3dVector(background_120[:, :3])
pcd_136.points = o3d.utility.Vector3dVector(background_136[:, :3])
pcd_139.points = o3d.utility.Vector3dVector(background_139[:, :3])

# 포인트 클라우드 시각화
o3d.visualization.draw_geometries([pcd_117])
o3d.visualization.draw_geometries([pcd_120])
o3d.visualization.draw_geometries([pcd_136])
o3d.visualization.draw_geometries([pcd_139])

