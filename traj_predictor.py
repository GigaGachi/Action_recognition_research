import cv2 
import numpy as np
from collections import deque
from time import time

import torch.utils
import torch.utils.data
import MCDWrapper
import pybgs as bgs
import torch
from torch import nn
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch


class kp_with_descr:
    def __init__(self, kp = (None,None), descr = ()):
        self.kp = kp
        self.descr = descr

class trajectory:
    def __init__(self,kp1 = None ,kp2 = None):
        self.trajectory_ = list()
        self.trajectory_.append(kp1)
        self.trajectory_.append(kp2)
        self.not_appended_for = 0
        self.size = 2
        self.last_kp = kp2
        self.speed_list = list()
        self.avg_speed = None
        self.real_size = 2
    def update(self,kp):
        self.trajectory_.append(kp)
        self.not_appended_for = 0
        self.size +=1
        self.last_kp = kp
        self.real_size+=1
    def update_with_null(self):
        self.trajectory_.append(kp_with_descr())
        self.not_appended_for+=1
        self.size +=1
    def dump_trajectory(self):
        print("[")
        for kp in self.trajectory_:
            print(kp.kp,"  ", end="")
        print("]")
    def calc_speed(self):
        p1 = self.trajectory_[-1].kp
        p2 = self.trajectory_[-2].kp
        
        if (p1 is not None) and (p2 is not None):
            speed_x = p2[0] - p1[0]
            speed_y = p2[1] - p1[1]
            self.speed_list.append((speed_x,speed_y))

    def remove_last_n_elements(self, arr,n):
            return arr[:-n] if n <= len(arr) else np.array([])

    def get_point_array(self):
        return [x.kp for x in self.trajectory_]
    
    def save_trajectory(self):
        print("trajectories/traj_" +f"{self.__hash__()}" +"_size_of_"+ f"{self.size}","trajectory surance:  ",self.real_size/self.size)
        # print(self.remove_last_n_elements(self.get_point_array(),10))
        np.save("trajectories/traj_" +f"{self.__hash__()}" +"_size_of_"+ f"{self.size}_{self.real_size/self.size:.3f}",np.array(self.remove_last_n_elements(self.get_point_array(),10)),allow_pickle=True)

    def analyze(self):
        
        delta_r = self.euclidean_distance(self.last_kp, self.trajectory_[0])
        # print(self.size,self.real_size/self.size,delta_r )
        if (self.size > 60) and ((self.real_size/self.size)>0.90) and ((delta_r)>40):
            
            self.save_trajectory()   
    
    def euclidean_distance(self,p1, p2):
            return np.sqrt((p1.kp[0] - p2.kp[0])**2 + (p1.kp[1] - p2.kp[1])**2)
        
    def calculate_trajectory_length(self,points):
        
        total_length = 0.0
        previous_point = None
        
        for point in points:
            if point:  # Check if the point is not an empty list
                if previous_point is not None:
                    total_length += self.euclidean_distance(previous_point, point)
                previous_point = point
        
        return total_length
    
class match:
    def __init__(self,kp1_with_descr,kp2_with_descr):
        self.first_point = kp1_with_descr
        self.second_point = kp2_with_descr

class trajectory_list:
    def __init__(self,overtime = 5):
        self.trajectory_list_ = list()
        self.size = 0
        self.overtime = overtime
        
    def update_trajectories(self,list_of_matches):
       
    #    For each match find the last keypoint occurence in trajectory list, then update or remove trajectories
    #    that were not updated for more them overtime frames, then save them trough analyze(sx)
                
        for match in list_of_matches:
            res = self.find_trajectory(match.first_point)
            if res is not None:
                curr_traj = self.trajectory_list_[res]
                curr_traj.update(match.second_point)
            else:
                # print(match.first_point)
                # print(match.second_point)
                self.trajectory_list_.append(trajectory(match.first_point,match.second_point))
                self.size+=1
        
        
        i = 0
        while i < self.size:
            curr_traj = self.trajectory_list_[i]
            if (curr_traj.not_appended_for == 0):
                curr_traj.not_appended_for+=1
                i+=1
                continue
            if (curr_traj.not_appended_for > self.overtime):
                curr_traj.analyze()
                self.trajectory_list_.pop(i)
                self.size-=1
                i = i -1
            else:
                curr_traj.update_with_null()
            i+=1
                                    
    def find_trajectory(self, kp):
        for i,trajectory in enumerate(self.trajectory_list_):
            if (trajectory.last_kp.kp == kp.kp):
                return i
        return None 
    
    def dump_trajectories(self):
        for i,trajectory in enumerate(self.trajectory_list_):
            print("trajectory ",  i ," of ", self.size, " size:  ", trajectory.size)
            # trajectory.dump_trajectory()
        
class traj_extractor:
    
    
    def __init__(self):
        self.dense_flow = FlowEstimator(winsize=25,levels=1)
        #self.sparse_tracker = cv.optflow.createOptFlow_SparseRLOF()
        self.sparse_tracker = cv2.calcOpticalFlowPyrLK
        self.optflow_params = dict(winSize=(25, 25), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))
        self.kp_descriptor  = cv2.xfeatures2d.TEBLID.create(6.25, cv2.xfeatures2d.TEBLID_SIZE_256_BITS)
        #self.kp_extractor = cv.ORB.create(2000, 2, 8, 38, 0, 2, cv.ORB_HARRIS_SCORE, 38, 20)
        #self.kp_extractor = cv2.goodFeaturesToTrack
        self.kp_extractor = cv2.AKAZE.create(threshold=0.001)
        self.good_features_params =  dict(maxCorners=100, qualityLevel=0.1, minDistance=25, blockSize=25)
        self.heatmap_sigma = 3
        self.current_kp = []
        self.prev_kp = []
        self.mcd = MCDWrapper.MCDWrapper()
        self.dynamic_points = deque(maxlen=2)
        self.frame_size = ()
        self.orig_frame = None
        self.cost_matrix = None
        self.overtime = 10
        self.debug = True
        self.trajectories = trajectory_list(self.overtime)
        self.euclidean_dist_weight = 1
        self.descr_dist_weight = 1
        self.cost_tresh = 50
        self.blob_detector_params = cv2.SimpleBlobDetector.Params()
        self.blob_detector_params.filterByCircularity = True
        self.blob_detector_params.minCircularity = 0.2
        self.blob_detector_params.maxCircularity = 1
        self.blob_detector_params.filterByConvexity = True
        self.blob_detector_params.minConvexity = 0.2
        self.blob_detector_params.maxConvexity = 1
        self.blob_detector_params.filterByInertia = True
        self.blob_detector_params.minInertiaRatio = 0.2
        self.blob_detector_params.maxInertiaRatio = 1
        self.blob_detector_params.filterByArea = False
        # self.blob_detector_params.maxArea = 500
        self.blob_detector_params.filterByColor = False
        # self.blob_detector_params.minThreshold = 100
        self.blob_detector = cv2.SimpleBlobDetector.create(self.blob_detector_params)
        
        self.bandwidth = 10
        self.treshold = 100
        self.resized = (640,512)
        
        #self.BGS = BGS_estimator()
          
    def keypoints_to_numpy(self, keypoints):
        return np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
    
    def create_heatmap_from_keypoints(self, keypoints):
        """
        Generate a heatmap from a list of keypoints.
        
        Parameters:
        - frame_size (tuple): Size of the frame (height, width).
        - keypoints (list): List of keypoints, each given as a tuple (x, y).
        - sigma (int): Standard deviation for Gaussian kernel.
        
        Returns:
        - heatmap (np.ndarray): Generated heatmap.
        """
        heatmap = np.zeros(self.frame_size, dtype=np.float32)
        height, width = self.frame_size
        
        for keypoint in keypoints:
            
            if (type(keypoint) == cv2.KeyPoint):
                x, y = keypoint.pt
            else:
                x,y = keypoint
            # Define the size of the region around the keypoint
            size = int(40)
            x_min = int(max(0, x - size))
            x_max = int(min(width, x + size + 1))
            y_min = int(max(0, y - size))
            y_max = int(min(height, y + size + 1))
            
            # Create a sub-region for the Gaussian mask
            sub_region = np.zeros((y_max - y_min, x_max - x_min), dtype=np.float32)
            
            # Coordinates of the keypoint in the sub-region
            sub_x = int(x - x_min)
            sub_y = int(y - y_min)
            
            # Apply Gaussian blur within the sub-region
            cv2.circle(sub_region,(sub_x, sub_y), 10, (1,), thickness=-1)
            sub_region = cv2.blur(sub_region,(15,15),None)
            
            # Add the sub-region to the heatmap
            heatmap[y_min:y_max, x_min:x_max] += sub_region
            
        
        # heatmap = cv2.normalize(heatmap, None, 0.0, 255.0, cv2.NORM_MINMAX)
        # print(heatmap)
        # heatmap = np.uint8(heatmap)
        # print()
    
        # Normalize the heatmap to the range [0, 255]
        # heatmap = cv2.normalize(heatmap, None, 0.0, 1.0, cv2.NORM_MINMAX)

        return heatmap
     
    def compute_hsv_image(self,magnitude,angle,heatmap):
        
        heatmap = cv2.normalize(heatmap,None,0,1,cv2.NORM_MINMAX,cv2.CV_8UC1)
        hsv = np.zeros((angle.shape[0], angle.shape[1], 3), dtype=np.float32)
        hsv[..., 0] = angle
        hsv[..., 1] = np.ones_like(angle)
        hsv[..., 2] = heatmap/2 + magnitude/2

        # Convert HSV to 8-bit
        hsv8 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        hsv8 = np.uint8(hsv8 * 255.0)
        
        return hsv8
        
    def compute_meanshift(self,image):
        mean_shift_result = cv2.pyrMeanShiftFiltering(image, sp=self.bandwidth, sr=self.bandwidth)

        # Convert the result to integer
        mean_shift_result = np.int8(mean_shift_result)
        
        return mean_shift_result
    
    def compute_dynamic_points(self, magnitude, angle, frame = None):
        """
        Compute dynamic keypoints using heatmap and optical flow.
        
        Parameters:
        - optflow (tuple): Optical flow (angle, magnitude).
        
        Returns:
        - keypoints (list): List of updated keypoints.
        """
        
        # Convert angle and magnitude to Cartesian coordinates
        # flow_x = magnitude * np.cos(angle*2*np.pi/360.)
        # flow_y = magnitude * np.sin(angle*2*np.pi/360.)
        
        
        

        # heatmap = cv2.normalize(heatmap,None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC3)
        # flow_x = cv2.normalize(flow_x,None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC1)
        # flow_y = cv2.normalize(flow_y,None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC1)
        # Create a composite image from heatmap and optical flow
        # composite_image = self.compute_hsv_image(magnitude,angle,heatmap)
        
        # Mean shift clustering
        
        
        # _,composite_image = cv2.threshold(composite_image,self.treshold,255,cv2.THRESH_TOZERO,None)
        # img_gray = cv2.cvtColor(composite_image,cv2.COLOR_BGR2GRAY)
        
        # img_gray = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
        # img_gray = img_gray.astype(np.uint8)
        
        # meanshift = self.compute_meanshift(composite_image)
        
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # fgmask = cv2.morphologyEx(composite_image, cv2.MORPH_DILATE, kernel)
        
        hsv = self.dense_flow.draw_flow(magnitude,angle)
        
        _,hsv = cv2.threshold(hsv,self.treshold,255,cv2.THRESH_TOZERO,None)
       
        hsv = hsv.astype(np.uint8)
        
        # hsv = self.mcd.run(frame)
        
        blob_centers = self.blob_detector.detect(hsv)
        
        
        
        blob_heatmap = self.create_heatmap_from_keypoints(blob_centers)
        
        
        # cv2.imshow("flow x",flow_x)
        # cv2.imshow("flow y",flow_y)
        
        # for y in range(0, composite_image.shape[0], 5):
        #     for x in range(0, composite_image.shape[1], 5):
        #         region = composite_image[max(0, y-2):min(composite_image.shape[0], y+3),
        #                                  max(0, x-2):min(composite_image.shape[1], x+3)]
                
        #         if np.any(region[..., 0] > 0):  # Heatmap intensity check
        #             region_mean = np.mean(region, axis=(0, 1))
        #             keypoints.append((x + region_mean[1], y + region_mean[2]))  # Weighted by flow
        # print(keypoints)
        
        cv2.imshow("blob heatmap",blob_heatmap)   
        cv2.imshow("flow",hsv)  
        # cv2.imshow("orig",self.orig_frame)  
        cv2.waitKey(1)
        # cv2.imshow("original",self.orig_frame) 
        return blob_centers

    def compute_dynamic_points_bgs(self, bgs_output, frame = None):
       
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        
        dialed_output = cv2.dilate(bgs_output,kernel,None)
        
        blob_centers = self.blob_detector.detect(dialed_output)
                
        blob_heatmap = self.create_heatmap_from_keypoints(blob_centers)
 
        if (self.debug):
            cv2.imshow("blob heatmap",blob_heatmap)   
            cv2.imshow("flow",dialed_output)  
            # cv2.imshow("orig",self.orig_frame)  
            cv2.waitKey(1)
        # cv2.imshow("original",self.orig_frame) 
        return blob_centers
  
    def test_create_trajectories_sparse(self, video_path):
        cap = cv2.VideoCapture(video_path)
        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        num_frames = 0
        old_frame = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
        old_frame = cv2.resize(old_frame,self.resized)
        self.mcd.init(old_frame)
        self.frame_size = [old_frame.shape[0],old_frame.shape[1]]
        with tqdm(total=1, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print('No frames grabbed!')
                    break
                frame = cv2.resize(frame,self.resized)
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                self.orig_frame = frame.copy()
                
                # heatmap = self.create_heatmap_from_keypoints(self.prev_kp)
                
                flow_t1 = time()
                # magn_norm, angle = self.dense_flow.run(old_frame,frame,None)
                #output = self.BGS.run_KNN(frame)
                output = self.mcd.run(frame)
                flow_t2 = time()
                
                kp_t1 = time()
                kp = self.compute_dynamic_points_bgs(output)
                #kp = self.compute_dynamic_points(magn_norm,angle,frame)
                kp_t2 = time()
                
                descr_t1 = time()
                descr = self.kp_descriptor.compute(frame,kp)
                
                
                self.append_dynamic_points(descr)
                descr_t2 = time()
                
                if (len(self.dynamic_points)>1):
                    
                    cost_t1 = time()
                    self.cost_matrix = self.compute_cost_matrix(self.dynamic_points)
                    cost_t2 = time()
                    # print(self.cost_matrix)
                    matches_t1 = time()
                    matches = self.compute_matches_from_cost_matrix(self.cost_matrix,self.dynamic_points)
                    matches_t2 = time()
                    
                    traj_update_t1= time()
                    self.trajectories.update_trajectories(matches)
                    traj_update_t2 = time()
                    

                    
                    if (self.debug):
                        draw_t1 = time()
                        self.draw_trajectories(self.orig_frame,self.trajectories)
                        draw_t2 = time()
                        self.trajectories.dump_trajectories()
                        print(f"FLOW TIME: {(flow_t2 - flow_t1) * 1000:.2f} ms")
                        print(f"DYNAMIC POINT TIME: {(kp_t2 - kp_t1) * 1000:.2f} ms")
                        print(f"DESCRIPTION TIME: {(descr_t2 - descr_t1) * 1000:.2f} ms")
                        print(f"COST MATRIX COMPUTE TIME: {(cost_t2 - cost_t1) * 1000:.2f} ms")
                        print(f"MATCHES TIME: {(matches_t2 - matches_t1) * 1000:.2f} ms")
                        print(f"TRAJECTORIES UPDATE TIME: {(traj_update_t2 - traj_update_t1) * 1000:.2f} ms")
                        print(f"DRAW TIME: {(draw_t2 - draw_t1) * 1000:.2f} ms")
                    
                # cv2.imshow('frame', flow)
                # k = cv2.waitKey(1) & 0xff
                # if k == 27:
                #     break

                old_frame = frame.copy()
                num_frames+=1

                pbar.update(1)    
                      
        cap.release()
        cv2.destroyAllWindows()
    
    def points_from_keypoints(self,list_of_kp):
        ret = list()
        for kp in list_of_kp:
            ret.append(kp.pt)
        return ret
    
    def append_dynamic_points(self,kp_with_descr):
        if (kp_with_descr[1] is None):
            return
        new_kp = self.points_from_keypoints(kp_with_descr[0])
        new_descr = list(kp_with_descr[1])
        self.dynamic_points.append(list([new_kp,new_descr]))

        for trajectory in self.trajectories.trajectory_list_:
            if (trajectory.not_appended_for > 1):
                self.dynamic_points[0][0].append(trajectory.last_kp.kp)
                self.dynamic_points[0][1].append(trajectory.last_kp.descr)
        
    def draw_trajectories(self,image, trajectories):
        i = 0
        while i < self.trajectories.size:
            # print(i,"  of  ", self.size)
            # random_color = (np.random.randint(64,128),np.random.randint(64,128), np.random.randint(64,128))
            random_color = (100,100,100)
            curr_traj = trajectories.trajectory_list_[i]
            if (curr_traj.size < 10):
                i+=1
                continue
            for j in range(1, curr_traj.size):
                # if either of the tracked points are None, ignore
                # them
                # print(curr_traj.trajectory_[j])
                if (curr_traj.trajectory_[j - 1].kp == None) or (curr_traj.trajectory_[j].kp == None):
                    continue
        
                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                thickness = 3
                cv2.line(image, curr_traj.trajectory_[j - 1].kp, curr_traj.trajectory_[j].kp, random_color, thickness)
            cv2.circle(image,curr_traj.last_kp.kp,2,random_color,-1)
            i+=1
        cv2.imshow("DRAW", image)
        cv2.waitKey(1)
    
    def compute_distance(self,kp1,descriptor1, kp2, descriptor2):
            # print(kp1,kp2)
            descr_dist = cv2.norm(descriptor1,descriptor2,cv2.NORM_HAMMING)
            euclid_dist = cv2.norm(kp1,kp2,cv2.NORM_L2)
            
            # print("descr: ",descr_dist)
            # print("euclid:  ",euclid_dist)
            # print()
            return descr_dist*self.descr_dist_weight  + euclid_dist*self.euclidean_dist_weight
        
    def compute_cost_matrix(self,dynamic_points):
        
        first_points = dynamic_points[0]
        second_points = dynamic_points[1]
        cost_matrix = np.zeros((len(first_points[0]),len(second_points[0])))
        for i in range(cost_matrix.shape[0] ):
            for j in range(cost_matrix.shape[1] ):
                
                kp1 = first_points[0][i]
                kp2 = second_points[0][j]
                descriptor1 = first_points[1][i]
                descriptor2 = second_points[1][j]
                cost_matrix[i][j] = self.compute_distance(kp1,descriptor1,kp2,descriptor2)
        # print(cost_matrix)
        return cost_matrix
    
    def compute_matches_from_cost_matrix(self,cost_matrix,dynamic_points):
        matches = list()
        indices_row = set()
        # indices = list()
        if (cost_matrix.size == 0):
            return matches
        for row in cost_matrix:
            minimum_row = np.argmin(row,0)
            indices_row.add(minimum_row)
            
        for i in indices_row:
            minimun_col = np.argmin(cost_matrix[:,i])
            if (cost_matrix[minimun_col,i] < self.cost_tresh):
            # indices.append([minimun_col,i])         
                # print(cost_matrix[minimun_col,i])   
                matches.append(match(kp_with_descr((int(dynamic_points[0][0][minimun_col][0]),int(dynamic_points[0][0][minimun_col][1])),
                                      dynamic_points[0][1][minimun_col])
                    ,kp_with_descr((int(dynamic_points[1][0][i][0]),int(dynamic_points[1][0][i][1])),
                      dynamic_points[1][1][i])))
            # print([minimun_col,i])
            # print((int(dynamic_points[0][0][minimun_col].pt[0]),int(dynamic_points[0][0][minimun_col].pt[1]))
            #       ,(int(dynamic_points[1][0][i].pt[0]),int(dynamic_points[1][0][i].pt[1])))
        # intersection = minimum_cols.intersection(minimum_rows)
        # print(intersection)
        # for match in intersection:
        #     matches.append(match)
        return matches
        
class BGS_estimator:
    def __init__(self):
        self.KNN = bgs.KNN()
        self.VIBE = bgs.ViBe()
        
    def run_KNN(self, img, debug=False):
        
        img_output = self.KNN.apply(img)
        
        return img_output

    def run_VIBE(self, img, debug=False):
        
        img_output = self.VIBE.apply(img)
        
        return img_output
    
class FlowEstimator:
    def __init__(self, pyr_scale=0.5, levels=3, winsize=21, iterations=3, poly_n=5, poly_sigma=1.2):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flow_flag = cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        self.DUAL_TVL1 = cv2.optflow.createOptFlow_DualTVL1()
        self.DUAL_TVL1.setInnerIterations(5)
        self.DUAL_TVL1.setOuterIterations(2)
        self.DUAL_TVL1.setEpsilon(0.1)
        self.DUAL_TVL1.setScalesNumber(1)
        self.DUAL_TVL1.setScaleStep(0.5)
        self.DUAL_TVL1.setLambda(0.3)
        self.DUAL_TVL1.setTau(0.5)
        
    def run(self, img1, img2, box = None, debug=False):
        
        if (box != None):
            x, y, w, h = box
            roi = (x, y, w, h)
            imgc1 = img1[y:y+h, x:x+w]
            imgc2 = img2[y:y+h, x:x+w]
        else:
            imgc1 = img1[:,:]
            imgc2 = img2[:,:]

        # image1 = cv2.cvtColor(imgc1, cv2.COLOR_BGR2GRAY)
        # image2 = cv2.cvtColor(imgc2, cv2.COLOR_BGR2GRAY)

        # blurred1 = cv2.GaussianBlur(image1, (5, 5), 2, 2)
        # blurred2 = cv2.GaussianBlur(image2, (5, 5), 2, 2)

        flow = cv2.calcOpticalFlowFarneback(
            imgc1, imgc2, None, 
            self.pyr_scale, self.levels, self.winsize, 
            self.iterations, self.poly_n, self.poly_sigma, 
            self.flow_flag
        )

        flow_parts = cv2.split(flow)
        magnitude, angle = cv2.cartToPolar(flow_parts[0], flow_parts[1], angleInDegrees=True)
        magn_norm = cv2.normalize(magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX)
        
        
        # magn_trsh = cv2.medianBlur(magn_norm, 3)
        
        return magn_norm, angle
    
    
    def run_rlof(self, img1, img2, box = None, debug=False):
        
        if (box != None):
            x, y, w, h = box
            roi = (x, y, w, h)
            imgc1 = img1[y:y+h, x:x+w]
            imgc2 = img2[y:y+h, x:x+w]
        else:
            imgc1 = img1[:,:]
            imgc2 = img2[:,:]

        # image1 = cv2.cvtColor(imgc1, cv2.COLOR_BGR2GRAY)
        # image2 = cv2.cvtColor(imgc2, cv2.COLOR_BGR2GRAY)

        # blurred1 = cv2.GaussianBlur(image1, (5, 5), 2, 2)
        # blurred2 = cv2.GaussianBlur(image2, (5, 5), 2, 2)

        flow = cv2.optflow.calcOpticalFlowDenseRLOF(imgc1, imgc2, None)
            

        flow_parts = cv2.split(flow)
        magnitude, angle = cv2.cartToPolar(flow_parts[0], flow_parts[1], angleInDegrees=True)
        magn_norm = cv2.normalize(magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX)
        
        
        # magn_trsh = cv2.medianBlur(magn_norm, 3)
        
        return magn_norm, angle
    
    
    
    def run_tvl(self, img1, img2, box = None, debug=False):
        
        if (box != None):
            x, y, w, h = box
            roi = (x, y, w, h)
            imgc1 = img1[y:y+h, x:x+w]
            imgc2 = img2[y:y+h, x:x+w]
        else:
            imgc1 = img1[:,:]
            imgc2 = img2[:,:]

        # image1 = cv2.cvtColor(imgc1, cv2.COLOR_BGR2GRAY)
        # image2 = cv2.cvtColor(imgc2, cv2.COLOR_BGR2GRAY)

        # blurred1 = cv2.GaussianBlur(image1, (5, 5), 2, 2)
        # blurred2 = cv2.GaussianBlur(image2, (5, 5), 2, 2)

        flow = self.DUAL_TVL1.calc(imgc1, imgc2, None)
            

        flow_parts = cv2.split(flow)
        magnitude, angle = cv2.cartToPolar(flow_parts[0], flow_parts[1], angleInDegrees=True)
        magn_norm = cv2.normalize(magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX)
        return magn_norm, angle
        
    def run_pca(self, img1, img2, box = None, debug=False):
        
        if (box != None):
            x, y, w, h = box
            roi = (x, y, w, h)
            imgc1 = img1[y:y+h, x:x+w]
            imgc2 = img2[y:y+h, x:x+w]
        else:
            imgc1 = img1[:,:]
            imgc2 = img2[:,:]

        # image1 = cv2.cvtColor(imgc1, cv2.COLOR_BGR2GRAY)
        # image2 = cv2.cvtColor(imgc2, cv2.COLOR_BGR2GRAY)

        # blurred1 = cv2.GaussianBlur(image1, (5, 5), 2, 2)
        # blurred2 = cv2.GaussianBlur(image2, (5, 5), 2, 2)

        flow = self.DUAL_TVL1.calc(imgc1, imgc2, None)
            

        flow_parts = cv2.split(flow)
        magnitude, angle = cv2.cartToPolar(flow_parts[0], flow_parts[1], angleInDegrees=True)
        magn_norm = cv2.normalize(magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX)
        
        # magn_trsh = cv2.medianBlur(magn_norm, 3)
        
        return magn_norm, angle
    
    def run_sf(self, img1, img2, box = None, debug=False):
        
        if (box != None):
            x, y, w, h = box
            roi = (x, y, w, h)
            imgc1 = img1[y:y+h, x:x+w]
            imgc2 = img2[y:y+h, x:x+w]
        else:
            imgc1 = img1[:,:]
            imgc2 = img2[:,:]

        # image1 = cv2.cvtColor(imgc1, cv2.COLOR_BGR2GRAY)
        # image2 = cv2.cvtColor(imgc2, cv2.COLOR_BGR2GRAY)

        # blurred1 = cv2.GaussianBlur(image1, (5, 5), 2, 2)
        # blurred2 = cv2.GaussianBlur(image2, (5, 5), 2, 2)

        flow = cv2.optflow.calcOpticalFlowSF(imgc1, imgc2,3,2,4)
            

        flow_parts = cv2.split(flow)
        magnitude, angle = cv2.cartToPolar(flow_parts[0], flow_parts[1], angleInDegrees=True)
        magn_norm = cv2.normalize(magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX)
        
        
        # magn_trsh = cv2.medianBlur(magn_norm, 3)
        
        return magn_norm, angle
    
    
    def draw_flow(self,magnitude,angle):
        hsv = np.zeros((angle.shape[0], angle.shape[1], 3), dtype=np.float32)
        hsv[..., 0] = angle
        hsv[..., 1] = np.ones_like(angle)
        hsv[..., 2] = magnitude

        # Convert HSV to 8-bit
        hsv8 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        hsv8 = np.uint8(hsv8 * 255.0)
        
        return hsv8
        
    
    def test(self,video_path):
        cap = cv2.VideoCapture(video_path)

        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        
        # old_frame = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
        
        
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print('No frames grabbed!')
                break

            magn_norm, angle = self.run(old_frame,frame,[0,0,1000,1000])
            flow = self.draw_flow(magn_norm,angle)
            img = (flow) #+ (frame/2)
            cv2.imshow('frame', img)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

            old_frame = frame
           
                    
            
        
        cap.release()
        cv2.destroyAllWindows()

class traj_predictor:
    
    def conv_layers_constructor(in_channels,out_dim,activation_function = nn.ReLU,num_layers = 1,kernel_size = 3,channel_scale = 2,dropout = 0):
    
        conv_layers = nn.Sequential()
        out_channels = int(in_channels*channel_scale)
        conv_layers.append(nn.Conv1d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    padding='same'))
        conv_layers.append(activation_function())
        conv_layers.append(nn.BatchNorm1d(out_channels))
        conv_layers.append(nn.Dropout1d(dropout))

        for layer in range(num_layers-2):
            in_channels = out_channels
            out_channels = int(in_channels*channel_scale)
            conv_layers.append(nn.Conv1d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        padding='same'))
            conv_layers.append(activation_function())
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.Dropout1d(dropout))
        in_channels = out_channels
        out_channels = int(out_dim)
        conv_layers.append(nn.Conv1d(in_channels=in_channels,
                                        out_channels=out_dim,
                                        kernel_size=kernel_size,
                                        padding='same'))
        conv_layers.append(activation_function())
        conv_layers.append(nn.BatchNorm1d(out_channels))
        conv_layers.append(nn.Dropout1d(dropout))    
        return conv_layers,out_channels

    def dense_layers_constructor(in_channels,activation_function = nn.ReLU,num_layers = 1,channel_scale = 2,dropout = 0):
        
        dense_layers = nn.Sequential()
        out_channels = int(in_channels*channel_scale)
        dense_layers.append(nn.Linear(in_features=in_channels,out_features=out_channels))
        dense_layers.append(activation_function())
        dense_layers.append(nn.BatchNorm1d(out_channels))
        dense_layers.append(nn.Dropout1d(dropout))
        
        for layer in range(num_layers-1):
            in_channels = out_channels
            out_channels = int(in_channels*channel_scale)
            dense_layers.append(nn.Linear(in_features=in_channels,out_features=out_channels))
            dense_layers.append(activation_function())
            dense_layers.append(nn.BatchNorm1d(out_channels))
            dense_layers.append(nn.Dropout1d(dropout))
            
        return dense_layers,out_channels

    def logit_layer(in_channels,out_channels):
        logit_layers = nn.Sequential()
        logit_layers.append(nn.Linear(in_channels,out_channels))
        logit_layers.append(nn.Softmax(1))
        return logit_layers


    def __init__(self,kernel_size = 3,dropout_conv = 0.3,
                 dropout_linear = 0.3,conv_out_dim = 14,
                 attention_heads = 4,attention_dropout = 0.3,
                 num_layers_conv = 2,channel_scale_conv = 2,
                 max_pool_size = 10,num_layers_dense = 2,
                 channel_scale_dense = 0.5):
        super().__init__()
        
        conv_layers,conv_channels = self.conv_layers_constructor(14,conv_out_dim,activation_function=nn.GELU,
                                                            num_layers=num_layers_conv,
                                                            kernel_size=kernel_size,
                                                            channel_scale=channel_scale_conv,
                                                            dropout=dropout_conv)
        self.conv_layers = conv_layers
        self.maxpool1 = nn.AdaptiveMaxPool1d(max_pool_size)

        self.attention = nn.LSTMCell(conv_channels ,256,batch_first=True,dropout= attention_dropout)

        self.flatten = nn.Flatten()
        
        dense_layers,dense_channels = self.dense_layers_constructor(conv_channels*max_pool_size,activation_function=nn.GELU,
                                                num_layers=num_layers_dense,
                                                channel_scale=channel_scale_dense,
                                                dropout = dropout_linear)
        self.linear = dense_layers
        self.output_layer = self.logit_layer(dense_channels,2)

        
    def forward(self, x):
        # forward pass сети
        x = self.conv_layers(x)
        x = self.maxpool1(x)
        
        x = torch.transpose(x,1,2)
        #x,_ = self.lstm1(x)
        x,_ = self.attention(x,x,x)

        x = self.flatten(x)
        x = self.linear(x)
        x = self.output_layer(x)
        
        return x

class npy_processor:
    def __init__(self,path):
        self.path_to_npy_directory = path
        self.list_of_sequence = None
    def load_data(self, path):
        npy_files = [f for f in os.listdir(path) if f.endswith('.npy')]
        arrays = []

        for npy_file in npy_files:
            file_path = os.path.join(path, npy_file)
            arrays.append(np.load(file_path,allow_pickle=True))
        
        return arrays
    
    def remove_last_n_elements(self, arr,n):
        return arr[:-n] if n <= len(arr) else np.array([])

    
    def process_data(self,list_of_sequence):
        for sequence in list_of_sequence:
            # sequence = self.remove_last_n_elements(sequence,10)
            # print(sequence)
            sequence = self.linear_interpolate(sequence)
            # print(sequence)
        return list_of_sequence
    
    def process_dataset(self,directory):
        
        list_of_sequence = self.load_data(directory)
        
        return self.process_data(list_of_sequence)
        
    def linear_interpolate(self, points):
        def interpolate(p1, p2, alpha):
            return [int(round(p1[i] * (1 - alpha) + p2[i] * alpha)) for i in range(len(p1))]
        
        n = len(points)
        result = points[:]
        
        for i in range(n):
            if points[i][0] is None:
                # Find previous and next valid points
                prev_idx = next_idx = None
                
                # Find previous valid point
                for j in range(i - 1, -1, -1):
                    if points[j][0] is not None:
                        prev_idx = j
                        break
                
                # Find next valid point
                for j in range(i + 1, n):
                    if points[j][0] is not None:
                        next_idx = j
                        break
                
                if prev_idx is not None and next_idx is not None:
                    # Interpolate between previous and next valid points
                    alpha = (i - prev_idx) / (next_idx - prev_idx)
                    result[i] = interpolate(points[prev_idx], points[next_idx], alpha)
                elif prev_idx is not None and next_idx is None:
                    # Use two previous valid points to interpolate
                    if prev_idx >= 1 and points[prev_idx - 1][0] is not None:
                        result[i] = interpolate(points[prev_idx - 1], points[prev_idx], 1)
                    else:
                        result[i] = points[prev_idx]
                elif prev_idx is None and next_idx is not None:
                    # Use two next valid points to interpolate
                    if next_idx + 1 < n and points[next_idx + 1][0] is not None:
                        result[i] = interpolate(points[next_idx], points[next_idx + 1], 0)
                    else:
                        result[i] = points[next_idx]
        
        return result

def normalize_2d_points(points,mean = None, std = None):
        """
        Normalize a sequence of 2D points.
        
        Parameters:
        points (np.ndarray): An array of shape (n, 2) where n is the number of points.
        
        Returns:
        normalized_points (np.ndarray): An array of normalized points of shape (n, 2).
        mean (np.ndarray): The mean of the original points.
        std (np.ndarray): The standard deviation of the original points.
        """
        points = np.array(points,dtype = np.float32)
        if (std is None and mean is None):
            mean = np.mean(points, axis=0)
            std = np.std(points, axis=0)

        for i,std_i in enumerate(std):
            if std_i < 0.01:
                std[i] = 0.00001

        
        normalized_points = (points - mean) / std
        return normalized_points, mean, std


class sequence_dataset(torch.utils.data.Dataset):
    
    def __init__(self, sequences, N, M):
        self.N = N
        self.M = M
        self.pairs = list()

        for seq in sequences:
            seq_len = len(seq)
            # Create pairs of smaller sequences of lengths M and N
            for i in range(0, seq_len - M - N + 1, N):
                sequence_M = seq[i:i + M]

                sequence_M,mean,std = normalize_2d_points(sequence_M)
                sequence_N = seq[i + M:i + M + N]
                
                sequence_N, _, __ = normalize_2d_points(sequence_N,mean,std)
                # print(mean,std)
                self.pairs.append((sequence_M, sequence_N,mean,std))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

class trajectory_predictor(nn.Module):
    def __init__(self, hidden_size, num_layers,input_seq_len,output_seq_len,dropout):
        super(trajectory_predictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = 2
        self.input_size = 2
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.lstm = nn.LSTM(2, hidden_size, num_layers, batch_first=True,dropout = dropout)
        self.fc = nn.Linear(hidden_size*input_seq_len, output_seq_len*2 )
        # self.normalize = nn.InstanceNorm1d(2)
        
    def forward(self, x):
        # print(x.shape)
        # print("input",x[0])
        # x = x.permute(0, 2, 1)
        # x = self.normalize(x)
        # x = x.permute(0, 2, 1)
        # print("normalized",x[0])
        # print(x.shape)
        lstm_out, _ = self.lstm(x)
        # print(lstm_out.shape)
        lstm_out = lstm_out.contiguous().view(lstm_out.size(0), -1)
        # print(lstm_out.shape)
        x = self.fc(lstm_out)
        # print(x.shape)
        x = x.view(-1, self.output_seq_len , self.output_size)
        # print(x.shape)
        return x

def train_best_model(config:dict):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(device)
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    input_seq_len= config['input_seq_len']
    output_seq_len = config['output_seq_len']
    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    dropout = config["dropout"]
    
    traj_path = "/home/iustimov/tasks/traj_pred/trajectories"

    npy_loader = npy_processor(traj_path)

    list_of_sequences = npy_loader.process_dataset(traj_path)

    train_data= sequence_dataset(list_of_sequences,output_seq_len,input_seq_len)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = trajectory_predictor(input_seq_len=input_seq_len,hidden_size=hidden_size,num_layers=num_layers,output_seq_len=output_seq_len,dropout=dropout)

    model.to(device)

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,amsgrad=False)

    trained_model = train_model(model=model,loss_fn=loss_fn,optimizer=optimizer,train_loader=train_loader,val_loader=train_loader,n_epoch=50,device=device)

    # loss_val = evaluate_model(trained_model,test_loader,loss_fn=loss_fn,device=device)

    return trained_model

def training_sequence(config: dict):
    
    split_percent = 0.75
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(device)
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    input_seq_len= config['input_seq_len']
    output_seq_len = config['output_seq_len']
    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    dropout = config["dropout"]
    
    traj_path = "/home/iustimov/tasks/traj_pred/trajectories"

    npy_loader = npy_processor(traj_path)

    list_of_sequences = npy_loader.process_dataset(traj_path)

    train_data= sequence_dataset(list_of_sequences,output_seq_len,input_seq_len)
    
    train_size = int(len(train_data) * split_percent)

    test_size = len(train_data) - train_size

    train_data, test_data= torch.utils.data.random_split(train_data, [train_size, test_size])
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    #val_loader = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = trajectory_predictor(input_seq_len=input_seq_len,hidden_size=hidden_size,num_layers=num_layers,output_seq_len=output_seq_len,dropout=dropout)

    model.to(device)

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,amsgrad=False)

    trained_model = train_model(model=model,loss_fn=loss_fn,optimizer=optimizer,train_loader=train_loader,val_loader=test_loader,n_epoch=50,device=device,raytune_mode=True)

    # loss_val = evaluate_model(trained_model,test_loader,loss_fn=loss_fn,device=device)

def evaluate_model(model, dataloader, loss_fn):
    
    losses = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model.train(False)
    for i, batch in enumerate(dataloader):

        with torch.no_grad():
            X_batch, y_batch, mean, std = batch 
                
            predicted = model(X_batch.to(device,dtype = torch.float)) 

            loss = loss_fn(predicted, y_batch.to(device,dtype = torch.float)) 
            
            losses+=loss
    
    
    return losses/len(dataloader.dataset)

def train_model(model,loss_fn, optimizer, train_loader: torch.utils.data.DataLoader,val_loader, device, n_epoch=3,raytune_mode = False):
    
        num_iter = 0
        # цикл обучения сети
        
        for epoch in range(n_epoch):
            
            model.train(True)
            for i, batch in enumerate(train_loader):
                # так получаем текущий батч
                
                X_batch, y_batch,mean,std = batch 
                # print("input",X_batch)
                # print("stats", mean, std)
                # print(y_batch.shape)
                # forward pass (получение ответов на батч картинок)
                predicted = model(X_batch.to(device,dtype = torch.float)) 
                # print("pred",predicted)
                # print("true",y_batch)
                # print(predicted.shape)
                # вычисление лосса от выданных сетью ответов и правильных ответов на батч
                loss = loss_fn(predicted, y_batch.to(device,dtype = torch.float)) 
                # print(loss)
                
                loss.backward() # backpropagation (вычисление градиентов)
                optimizer.step() # обновление весов сети
                optimizer.zero_grad() # обнуляем веса

                num_iter += 1


            # после каждой эпохи получаем метрику качества на валидационной выборке
            if ((epoch+1)%10 ==0) and (raytune_mode == True):
                model.train(False)
                val_loss = evaluate_model(model, val_loader, loss_fn=loss_fn)
                # train_loss = evaluate_model(model, train_loader, loss_fn=loss_fn)
                torch.save((model.state_dict(), optimizer.state_dict()), "/home/iustimov/tasks/traj_pred/checkpoint_models/checkpoint.pt")
                checkpoint = Checkpoint.from_directory("/home/iustimov/tasks/traj_pred/checkpoint_models")
                train.report({"loss": val_loss.cpu().numpy()}, checkpoint=checkpoint)
       
            # if ((epoch+1)%25 ==0) and (raytune_mode == True):
            #     os.makedirs("checkpoint_models", exist_ok=True)
            #     torch.save(
            #                 (model.state_dict(), optimizer.state_dict()), "checkpoint_models/checkpoint.pt")
            #     checkpoint = Checkpoint.from_directory("checkpoint_models")
            #     train.report({"loss": val_loss}, checkpoint=checkpoint)

        return model

def ray_tune(config_hp,num_samples=1, max_num_epochs=100, gpus_per_trial=1):
    
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(training_sequence),
            resources={"cpu": 10, "gpu": gpus_per_trial}
        ),
        param_space=config_hp,
        run_config=train.RunConfig(
        name="traj-exp",
        storage_path="/home/iustimov/tasks/traj_pred/ray/",
        checkpoint_config=train.CheckpointConfig(
            checkpoint_score_attribute="loss",
            checkpoint_score_order="min",
            num_to_keep=5,
        ),
        ),
        tune_config=tune.TuneConfig(
            search_alg=OptunaSearch(),
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        
    )

    results = tuner.fit()
        
    best_result = results.get_best_result("loss", "min")
    
    print("Best trial config: {}".format(best_result.config))
        
    best_model = train_best_model(best_result.config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    example_input = torch.randn((1, best_result.config["input_seq_len"], 2)).to(device=device)
    onnx_file_path = "trajectory_predictor.onnx"
    torch.onnx.export(best_model, 
                  example_input, 
                  onnx_file_path, 
                  export_params=True, 
                  opset_version=10, 
                  do_constant_folding=True, 
                  input_names=['input'], 
                  output_names=['output'], 
                  )

    print(f"Model has been exported to {onnx_file_path}")
        
    return results