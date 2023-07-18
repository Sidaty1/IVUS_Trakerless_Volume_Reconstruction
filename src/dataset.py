from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from parameters import *
from utils import *

import cv2
import os
import numpy as np 
import torch


class Data(Dataset): 
    """

    """
    def __init__(self, img_dim=(128, 128), k=2, type="train") -> None:
        self.data = []
        self.count_augmented_frames = 1
        self.count_augmented_videos = 1
        for scan in scans: 
            pth_track = "./data/"+ type + "/" + scan + "tracking/"
            pth_frames = "./data/"+ type + "/" + scan + "frames/"
            assert len(os.listdir(pth_track)) == len(os.listdir(pth_frames)), f"Directory {pth_frames} and {pth_track} should contain the same number of elements."
            start = get_start_point(os.listdir(pth_track))
            for i in range(1, len(os.listdir(pth_track))):
                pth_vid = pth_frames + str(start + i) + '.npy'
                pth_trk= pth_track + str(start + i) + '.npy'
                if self.sanity_check(pth_trk):
                    self.data.append([pth_vid, pth_trk])

        self.img_dim = img_dim
        self.k = k

            
    def sanity_check(self, trk_pth):
        vid_trk = np.load(trk_pth)
        trk0 = matrix_to_rot_and_trans(np.load(vid_trk[0]))[0].as_euler("xyz", degrees=True)
        trk = matrix_to_rot_and_trans(np.load(vid_trk[1]))[0].as_euler("xyz", degrees=True)
        tmp = np.array(trk) - np.array(trk0)
        for i in range(0, len(trk)-1):
            trk0 = matrix_to_rot_and_trans(np.load(vid_trk[i]))[0].as_euler("xyz", degrees=True)
            trk = matrix_to_rot_and_trans(np.load(vid_trk[i+1]))[0].as_euler("xyz", degrees=True)
            rlv_trk = np.array(trk) - np.array(trk0)
            if rlv_trk[0] * tmp[0] < 0: 
                return False
        return True   
        
    def __len__(self):
        return len(self.data)
    
    def _get_traj_matrix_(self, frames):
        prevgray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(prevgray, mask = None, **feature_params)
        if isinstance(p0, np.ndarray):
          Vectors = [np.asarray([pt for pt in p0])]
        else: 
          Vectors = []
        for j in range(1, len(frames)):
            frame = frames[j]
            vector = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for point in p0: 
                if point[0][0] == 0. and point[0][1] == 0.:
                    vector.append([[0., 0.]])
                else:
                    if not isinstance(point, np.ndarray): 
                        point = np.asarray(point)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(prevgray, gray, point, None, **lk_params)
                    if st[0, 0] == 1:
                        vector.append(p1)
                    elif st[0, 0] == 0:
                        vector.append([[0., 0.]])

            new_points = cv2.goodFeaturesToTrack(gray, mask = None, **feature_params)
            if isinstance(new_points, np.ndarray):
                for pt in new_points:
                    vector.append(pt)
            
            Vectors.append(np.asarray(vector))
            p0 = vector
            prevgray = gray.copy()
            
        Vectors = np.asarray(Vectors, dtype=object)
        mat = np.zeros((Vectors[-1].shape[0], len(frames), 2))
        for i in range(-1, -mat.shape[1] - 1, -1):
            mat[:Vectors[i].shape[0],i,:] = np.squeeze(Vectors[i])
        return mat

    def _gaussian_heatmap_(self, sigma, position, size):
        heatmap = np.zeros(size)
        if not np.array_equal(position,[[-1., -1.], [-1., -1]]):
          for i_ in range(size[0]):
              for j_ in range(size[1]):
                  heatmap[i_, j_] = 1 / 2 / np.pi / (sigma ** 2) * np.exp(
                      -1 / 2 * ((i_ - position[0] - 0.5) ** 2 + (j_ - position[1] - 0.5) ** 2) / (sigma ** 2))
          heatmap = (heatmap / np.max(heatmap) * 255)
        return heatmap

    def _list_index_tracking_points(self, frame0, frame1): 
        indxs = []
        assert frame0.shape[0] == frame1.shape[0], "Frames tracking should have the same shape"   
        for i in range(frame1.shape[0]):
            if not all(frame0[i] == frame1[i]): 
                indxs.append(i)
        return indxs

    def _get_k_tracking_points(self,  frame0, frame1, k): 
        trackings = []
        card = len(self._list_index_tracking_points(frame0, frame1))
        if k <= card: 
            for i in range(k):
                trackings.append([frame0[i], frame1[i]])
        else: 
            nb_trk = k - card
            for i in range(nb_trk): 
                trackings.append([frame0[i], frame1[i]])
            
            for i in range(nb_trk, k): 
                trackings.append([[-1., -1.], [-1, -1]])

        return trackings            

    def _get_heatmaps(self, trackings, size, std): 
        heatmap = np.zeros((len(trackings), 2, *size))
        
        for i, tracking in enumerate(trackings): 
            heatmap0 = self._gaussian_heatmap_(std, position=tracking[0], size=size)
            heatmap1 = self._gaussian_heatmap_(std, position=tracking[1], size=size)

            heatmap[i, 0] = heatmap0
            heatmap[i, 1] = heatmap1

        return heatmap

    def __getitem__(self, index):
        img_path, track_path = self.data[index]

        img_path_ = np.load(img_path)
        track_path = np.load(track_path)
        video = []
        for pth in img_path_: 
            img = cv2.imread(pth)

            # removing the description on the US images, the numbers were experimentaly found 
            img = img[20:440,60:600] 

            img = cv2.resize(img, self.img_dim)

            video.append(img)
        
        trj_mat = self._get_traj_matrix_(video)

        three_frames_trj = np.zeros((trj_mat.shape[0], 3, 2))
        three_frames_trj[:, 0] = trj_mat[:,0]
        three_frames_trj[:, 1] = trj_mat[:,2]
        three_frames_trj[:, 2] = trj_mat[:,4]

        trackings_0_to_1 = self._get_k_tracking_points(three_frames_trj[:, 0], three_frames_trj[:, 1], self.k)
        trackings_1_to_2 = self._get_k_tracking_points(three_frames_trj[:, 1], three_frames_trj[:, 2], self.k)
        heatmaps = np.zeros((2, self.k+1, 2, *self.img_dim))

        img0 = cv2.cvtColor(video[0], cv2.COLOR_BGR2GRAY)
        img3 = cv2.cvtColor(video[2], cv2.COLOR_BGR2GRAY)
        img6 = cv2.cvtColor(video[4], cv2.COLOR_BGR2GRAY)
        

        img0 = 255*(img0 - img0.min())/(img0.max() - img0.min())
        img3 = 255*(img3 - img3.min())/(img3.max() - img3.min())
        img6 = 255*(img6 - img6.min())/(img6.max() - img6.min())
        
        heatmaps[0, 0] = [img0, img3]
        heatmaps[0, 1:] = self._get_heatmaps(trackings_0_to_1, size=self.img_dim, std=5)

        heatmaps[1, 0] = [img3, img6]
        heatmaps[1, 1:] = self._get_heatmaps(trackings_1_to_2, size=self.img_dim, std=5)


        heatmaps = torch.from_numpy(heatmaps)
        trackings = np.zeros((2, 6))

        src_track = np.load(track_path[0])
        trg_track = np.load(track_path[2])
        rot_src, t_src = matrix_to_rot_and_trans(src_track)
        rot_trg, t_trg = matrix_to_rot_and_trans(trg_track)
        angles_src = rot_src.as_euler("xyz", degrees=True)
        angles_trg = rot_trg.as_euler("xyz", degrees=True)
        t_relative = t_trg - t_src
        angles = angles_trg - angles_src
        for i in range(len(angles)): 
            if angles[i] > 180: 
                angles[i] = angles[i] - 360
            if angles[i] < -180: 
                angles[i] = angles[i] + 360
                
        src_track = np.load(track_path[2])
        trg_track = np.load(track_path[4])
        rot_src, t_src = matrix_to_rot_and_trans(src_track)
        rot_trg, t_trg = matrix_to_rot_and_trans(trg_track)
        angles_src = rot_src.as_euler("xyz", degrees=True)
        angles_trg = rot_trg.as_euler("xyz", degrees=True)
        t_relative = t_trg - t_src
        angles = angles_trg - angles_src
        for i in range(len(angles)): 
            if angles[i] > 180: 
                angles[i] = angles[i] - 360
            if angles[i] < -180: 
                angles[i] = angles[i] + 360
        trackings[1] = [x for x in t_relative] + [angle  for angle in angles]
        
        track_tensor = torch.from_numpy(trackings)
        return heatmaps, track_tensor
