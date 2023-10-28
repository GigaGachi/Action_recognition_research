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

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device  = torch.device('cpu')
print(device)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = YOLO('yolov8s-pose.pt')  # load a pretrained YOLOv8n classification model
model.to(device)
video_path = r"D:\videos\stul.mp4"
cap = cv2.VideoCapture(video_path)
# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS) # or number
# Create a VideoWriter object to save the output video
output_video_path = r"D:\videos_processed\stul_processed.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def extract_keypoints(results, threshold_class):
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
        dist = int(np.linalg.norm(p1-p2,ord = 2))
        return dist
    else: 
        return None
    
def calc_angle(v1,v2):
    if (len(v1)>0) and (len(v2)>0):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return 360*(np.arccos(cos_angle))/(2*math.pi)
    else: 
        return None

def calc_angles(keypoints_dict,kp_conf):
    #angles dictionary to store angles between doby and legs of each object
    angles_dic = {}
    for key in keypoints_dict:
        #extracting keypoints
        left_shoulder = keypoints_dict[key][5]
        if left_shoulder[2] < kp_conf:
            angles_dic[f'{key}'] = [180,180]
            continue
        right_shoulder = keypoints_dict[key][6]
        if right_shoulder[2] < kp_conf:
            angles_dic[f'{key}'] = [180,180]
            continue
        left_hip = keypoints_dict[key][11]
        if left_hip[2] < kp_conf:
            angles_dic[f'{key}'] = [180,180]
            continue
        right_hip = keypoints_dict[key][12]
        if right_hip[2] < kp_conf:
            angles_dic[f'{key}'] = [180,180]
            continue
        left_knee = keypoints_dict[key][13]
        if left_knee[2] < kp_conf:
            angles_dic[f'{key}'] = [180,180]
            continue
        right_knee = keypoints_dict[key][14]
        if right_knee[2] < kp_conf:
            angles_dic[f'{key}'] = [180,180]
            continue
        #calculating vectors between keypoints
        vl1 = [left_knee[0]-left_hip[0],left_knee[1]-left_hip[1]]
        vl2 = [left_shoulder[0] - left_hip[0], left_shoulder[1] -left_hip[1] ]
        vr1 = [right_knee[0]-right_hip[0],right_knee[1]-right_hip[1]]
        vr2 = [right_shoulder[0] - right_hip[0], right_shoulder[1] - right_hip[1]]
        #calculating angles
        angll = calc_angle(vl1,vl2)
        anglr = calc_angle(vr1,vr2)
        angles_dic[f'{key}'] = [angll,anglr]
    return angles_dic

def annotate_object(box,box_color,text,font_scale,font_thickness,frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Calculate the position to align the label with the top of the bounding box
    text_x = x1 + (x2 - x1 - text_size[0]) // 2
    text_y = y1 - 10  # Adjust this value for the desired vertical offse
    # Make sure the text_y position is within the frame's bounds
    if text_y < 0:
        text_y = 0
     # Draw the label background rectangle
    cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)

     #Draw the customer id text
    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),  # White color
        font_thickness,
        lineType=cv2.LINE_AA
        )

def calc_grad(dist_dict):

    return

text2 = "No sitting"
text1 = "Sitting"
text3 = "No people in sight"
color2 = (100, 200, 0)
color1 = (100, 0, 200)
color3 = (100, 100, 100)
font_scale = 1.6
thickness = 2

winsize = 60
angles_dict = {}
angle_tresh = 112
all_keypoints = {}
average_angles = {}
grad_dict = {}
while cap.isOpened():
# Read a frame from the video
    success, frame = cap.read()
    if success:
        frame = cv2.resize(frame,(1280,704))
        results = model.track(frame, persist=True, retina_masks=True, boxes=True, show_conf=False, line_width=1,  conf=0.3, iou=0.5,  classes=0, show_labels=False, device=device,verbose = True,tracker="bytetrack.yaml")
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            #extracting keypoints

            kp = extract_keypoints(results = results,threshold_class=0.2)
    
            #calculating distances between keypoints

            angles = calc_angles(kp,0.5)
            #appending distances dictionary and evaluating average distance and classification based on it
            for key in angles.keys():

                if key not in angles_dict.keys():
                    angles_dict[key] = deque(maxlen=winsize)

                angles_dict[key].append(angles[key])
                average_angles[key] = np.mean(angles_dict[key])
                for angle in angles[key]:
                    aydi = np.where(ids == int(key))[0]
                    if 70<angle < angle_tresh:
                        annotate_object(boxes[aydi][0],(255,0,0),"Person is sitting",1,1,frame)
                        break
                    else: 
                        annotate_object(boxes[aydi][0],(0,255,0),"Person is not sitting",1,1,frame)


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


