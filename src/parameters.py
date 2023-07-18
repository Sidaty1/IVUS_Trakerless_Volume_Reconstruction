import cv2
import numpy as np

# data parameters
scans = [
     "init/", 
]

# network parameters
num_channels=3
in_channel=1
out_channel=3
padding=1
activation='tanh'
frame_size=(128, 128)
num_layers=8
features = [8192, 2048, 1024, 256, 8]


# training parameters
epochs=100
min_valid_loss=np.inf 
batch_size=16
lr=0.0001
T_max=10
eta_min=0
feature_params = dict( maxCorners=500,
                       qualityLevel=0.1,
                       minDistance=4,
                       blockSize = 4)
lk_params = dict( winSize=(15,15),
                  maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

train_loss = 0.0
train_mse = 0.0
train_cycle = 0.0
valid_loss = 0.0
valid_mse = 0.0
valid_cycle = 0.0