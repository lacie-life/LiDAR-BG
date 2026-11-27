import os
import shutil
import natsort
from pypcd4 import PointCloud
import numpy as np


file_names = natsort.natsorted(os.listdir("./intensity_normalized_64ch_north_south_r0_pcd"))

for file_name in file_names:
    
    print(os.path.join("./intensity_normalized_64ch_north_south_r0_pcd",file_name))
    np_pc = PointCloud.from_path(os.path.join("./intensity_normalized_64ch_north_south_r0_pcd",file_name)).numpy().astype(np.float32)
    np_pc = np_pc[:,:4]
    np_pc.tofile(os.path.join("intensity_normalized_64ch_north_south_r0_pcd_bin",file_name[:-4]+".bin"))
    