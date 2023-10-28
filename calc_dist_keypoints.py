from ultralytics import YOLO
from pydantic import BaseModel
import cv2
import torch
import os
import time
import numpy as np
import math
from collections import deque

keypoints_dict = {
    0: 'NOSE',
    1: 'LEFT_EYE',
    2: 'RIGHT_EYE',
    3: 'LEFT_EAR',
    4: 'RIGHT_EAR',
    5: 'LEFT_SHOULDER',
    6: 'RIGHT_SHOULDER',
    7: 'LEFT_ELBOW',
    8: 'RIGHT_ELBOW',
    9: 'LEFT_WRIST',
    10: 'RIGHT_WRIST',
    11: 'LEFT_HIP',
    12: 'RIGHT_HIP',
    13: 'LEFT_KNEE',
    14: 'RIGHT_KNEE',
    15: 'LEFT_ANKLE',
    16: 'RIGHT_ANKLE'
}

# Define colors for each group
head_color = (0, 255, 0)  # Green
hands_and_shoulders_color = (255, 0, 0)  # Blue
body_color = (128, 0, 128)  # Purple
hips_and_feet_color = (0, 165, 255)  # Orange
connections = [
    ('NOSE', 'LEFT_EYE', head_color),
    ('NOSE', 'RIGHT_EYE', head_color),
    ('LEFT_EYE', 'RIGHT_EYE', head_color),
    ('LEFT_EYE', 'LEFT_EAR', head_color),
    ('RIGHT_EYE', 'RIGHT_EAR', head_color),
    ('LEFT_EAR', 'LEFT_SHOULDER', head_color),
    ('RIGHT_EAR', 'RIGHT_SHOULDER', head_color),
    ('LEFT_SHOULDER', 'RIGHT_SHOULDER', hands_and_shoulders_color),
    ('LEFT_SHOULDER', 'LEFT_ELBOW', hands_and_shoulders_color),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW', hands_and_shoulders_color),
    ('LEFT_ELBOW', 'LEFT_WRIST', hands_and_shoulders_color),
    ('RIGHT_ELBOW', 'RIGHT_WRIST', hands_and_shoulders_color),
    ('LEFT_SHOULDER', 'LEFT_HIP', body_color),
    ('RIGHT_SHOULDER', 'RIGHT_HIP', body_color),
    ('LEFT_HIP', 'RIGHT_HIP', body_color),
    ('LEFT_HIP', 'LEFT_KNEE', hips_and_feet_color),
    ('RIGHT_HIP', 'RIGHT_KNEE', hips_and_feet_color),
    ('LEFT_KNEE', 'LEFT_ANKLE', hips_and_feet_color),
    ('RIGHT_KNEE', 'RIGHT_ANKLE', hips_and_feet_color),
]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = YOLO('yolov8x-pose.pt')  # load a pretrained YOLOv8n classification model
model.to(device)
video_path = r"D:\videos\hands3.mp4"
cap = cv2.VideoCapture(video_path)
# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS) # or number
# Create a VideoWriter object to save the output video
output_video_path = r"D:\videos_processed\hands3_processed.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def extract_keypoints(results, threshold_class, threshold_keypoint):
    existing_kp = {}
    for result,i_d in zip(results[0],results[0].boxes.id):
        # There results for bounding boxes, and confidence scores for general detect
        x1, y1, x2, y2,_, conf_for_detect, class_id_detected = (result.boxes.data.tolist())[0]
        # If the confidence score for general detect is lower than threshold, skip
        if conf_for_detect < threshold_class:
            continue
        # keypoints
        keys = (result.keypoints.data.tolist())[0]
        keyp_arr = list()
        for key in keys:
            keyp_arr.append(key)
        # Adding existing hand keypoints of an object in a frame to the dictionary   
        existing_kp[int(i_d)] = keyp_arr
    return existing_kp

def calc_kp_to_kp_dist(keypoints_dict):
    # creating a dictionary of distances between each keypoint (except of the same object) in the keypoint_dict
    dist_dict = {}
    num_obj = len(keypoints_dict.keys())
    keys = keypoints_dict.keys()
    # calculating distances between keypoints 
    for l,keyi in enumerate(keys,start =1):
        for m,keyj in enumerate(keys,start =1):
            if m>=l:
                break  
            for i,p1 in enumerate(keypoints_dict[keyi]):
                for j,p2 in enumerate(keypoints_dict[keyj]):
                    dist = calc_euclid_dist(p1,p2)
                    dist_dict[f'{keyi}'+f'{keyj}'+f'{i}'+f'{j}'] = dist
    return dist_dict

def calc_euclid_dist(p1,p2):
    if (len(p1)>0) and (len(p2)>0):
        dist = int(math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1])))
        return dist
    else: 
        return None
    
def calc_grad(dist_dict):
    return

text2 = "No suspicious activity"
text1 = "Suspicious activity"
text3 = "No people in sight"
color2 = (100, 200, 0)
color1 = (100, 0, 200)
color3 = (100, 100, 100)
font_scale = 1.6
thickness = 2

winsize = 60
all_keypoints = {}
distance_dict = {}
average_dist = {}
grad_dict = {}
while cap.isOpened():
# Read a frame from the video
    success, frame = cap.read()
    if success:

        results = model.track(frame, persist=True, retina_masks=True, boxes=True, show_conf=False, line_width=1,  conf=0.3, iou=0.5,  classes=0, show_labels=False, device=device,verbose = False,tracker="bytetrack.yaml")
        text_size, _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_position = (frame_width - text_size[0] - 10, text_size[1] + 10)
        cv2.rectangle(frame, (text_position[0] - 5, text_position[1] - text_size[1] - 5),
                                    (text_position[0] + text_size[0] + 5, text_position[1] + 5), color=(0, 0, 0),
                                    thickness=cv2.FILLED)
        cv2.putText(frame, text2, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color2, thickness, cv2.LINE_AA)
        if results[0].boxes.id is not None:

            
            
            #extracting keypoints
            kp = extract_keypoints(results = results,threshold_class=0.2,threshold_keypoint=0.2)
    
            #appending keypoints to dictionary with size = winsize frames window

            for i_d in results[0].boxes.id:
                if int(i_d) not in all_keypoints.keys():
                    all_keypoints[int(i_d)] = deque(maxlen=winsize)
                all_keypoints[int(i_d)].append(kp[int(i_d)])

            #calculating distances between keypoints

            dd = calc_kp_to_kp_dist(kp)
            #appending distances dictionary and evaluating average distance and classification based on it
            for key in dd.keys():

                if key not in distance_dict.keys():
                    distance_dict[key] = deque(maxlen=winsize)

                average_dist[key] = np.mean(distance_dict[key])
                distance_dict[key].append(dd[key])

                if dd[key]< average_dist[key]/1.8:
                    text_size, _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    text_position = (frame_width - text_size[0] - 10, text_size[1] + 10)
                    cv2.rectangle(frame, (text_position[0] - 5, text_position[1] - text_size[1] - 5),
                                    (text_position[0] + text_size[0] + 5, text_position[1] + 5), color=(0, 0, 0),
                                    thickness=cv2.FILLED)
                    cv2.putText(frame, text1, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color1, thickness, cv2.LINE_AA)
                    continue


        annotated_frame_show = cv2.resize(frame, (1080, 720))
        cv2.imshow("YOLOv8 Inference", annotated_frame_show)
        out.write(frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        
    else:
        # Break the loop if the end of the video is reached
        break
cap.release()
out.release()
cv2.destroyAllWindows()


