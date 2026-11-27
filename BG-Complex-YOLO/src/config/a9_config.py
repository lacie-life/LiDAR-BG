import numpy as np

class_list = ['CAR', 'WHEELER', 'PEDESTRIAN']

# class_list = ['CAR', 'TRUCK','TRAILER','VAN','MOTORCYCLE','BUS','PEDESTRIAN','BICYCLE','EMERGENCY_VEHICLE','OTHER']

CLASS_NAME_TO_ID = {
    'CAR': 0,
    'WHEELER': 1,
    'PEDESTRIAN': 2
}

# CLASS_NAME_TO_ID = {
#     'CAR': 0,
#     'TRUCK' : 1,
#     'TRAILER' : 2,
#     'VAN' : 3,
#     'MOTORCYCLE' : 4,
#     'BUS' : 5,
#     'PEDESTRIAN' : 6,
#     'BICYCLE' : 7,
#     'EMERGENCY_VEHICLE' : 8,
#     'OTHER' : 9
# }

# Front side (of vehicle) Point Cloud boundary for BEV
boundary = {
    "minX": 0,
    "maxX": 100,
    "minY": -50,
    "maxY": 50,
    "minZ": -10,
    "maxZ": -2.56
    # "maxZ": -5
}

# boundary = {
#     "minX": 0,
#     "maxX": 50,
#     "minY": -25,
#     "maxY": 25,
#     "minZ": -10,
#     "maxZ": -2.56
#     # "maxZ": -5
# } 

# Back back (of vehicle) Point Cloud boundary for BEV
boundary_back = {
    "minX": -40,
    "maxX": 0,
    "minY": -40,
    "maxY": 0,
    "minZ": -9,
    "maxZ": 0
}
BEV_WIDTH = 608  # across y axis 0m ~ 100m
BEV_HEIGHT = 608  # across x axis -50m ~ 50m

DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT

# colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]]
colors = [[255, 255, 255], [128, 128, 128], [0, 255, 255]]
gt_colors = [[255,0, 0], [255, 0, 255], [0, 0, 255]]

# Following parameters are calculated as an average from A9 dataset for simplicity
#####################################################################################
# cam to lidar to base
cam_to_lidars = np.array(
    [                        0.641509,
                        -0.766975,
                        0.0146997,
                        1.99131,
                        -0.258939,
                        -0.234538,
                        -0.936986,
                        1.21464,
                        0.722092,
                        0.597278,
                        -0.349058,
                        -1.50021,
                        0.0,
                        0.0,
                        0.0,
                        1.0]

)

lidar_to_base = np.array([
    [0.99011437,-0.13753536,-0.02752358,4.81100903],
    [0.13828977,0.99000475,0.02768645,-15.94524012],
    [0.02344061,-0.03121898,0.99923766,-8.05225519],
    [0, 0, 0, 1]
])

lidar_to_base_real = np.array([
    [0.21479485, -0.9761028, 0.03296187, -15.87257873],
    [0.97627128, 0.21553835, 0.02091894, 2.30019086],
    [-0.02752358, 0.02768645, 0.99923767, 7.48077521],
    [0.00000000, 0.00000000, 0.00000000, 1.00000000]
  ]
)

lidar_to_base_real_real = np.array([
                [0.95812397, -0.28634765, -0.00186487, 1.39543971],
                [0.28633102, 0.95810969, -0.00635344, -13.9356065],
                [0.00360605, 0.00555341, 0.99997808, 0.1595251],
                [0.0, 0.0, 0.0, 1.0],
  ]
)


Tr_velo_to_cam = np.array([
    [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
    [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
    [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
    [0, 0, 0, 1]
])

# cal mean from train set
R0 = np.array([
    [0.99992475, 0.00975976, -0.00734152, 0],
    [-0.0097913, 0.99994262, -0.00430371, 0],
    [0.00729911, 0.0043753, 0.99996319, 0],
    [0, 0, 0, 1]
])

P2 = np.array([[719.787081, 0., 608.463003, 44.9538775],
               [0., 719.787081, 174.545111, 0.1066855],
               [0., 0., 1., 3.0106472e-03],
               [0., 0., 0., 0]
               ])

R0_inv = np.linalg.inv(R0)
Tr_velo_to_cam_inv = np.linalg.inv(Tr_velo_to_cam)
P2_inv = np.linalg.pinv(P2)
#####################################################################################
