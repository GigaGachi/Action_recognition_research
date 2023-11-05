from ultralytics import YOLO
import cv2
import torch
import os
import numpy as np
from math import sqrt
from collections import deque
from time import time

def calc_distances(hands_dict,body_kp,head_kp):

    # creating a dictionary of distances between each keypoint (except of the same object) in the keypoint_dict
    dist_dict = {}
    keyhead = head_kp.keys()
    keysh = hands_dict.keys()
    keysb = body_kp.keys()

    # calculating distances between keypoints on each hand: left to left, right to left, left to right and right to right
    for i,keyi in enumerate(keysh,start =1):
        for j,keyj in enumerate(keysh,start =1):
            if j>=i:
                break
            distll = calc_euclid_dist(hands_dict[keyi][0],hands_dict[keyj][0])
            distlr = calc_euclid_dist(hands_dict[keyi][0],hands_dict[keyj][1])
            distrl = calc_euclid_dist(hands_dict[keyi][1],hands_dict[keyj][0])
            distrr = calc_euclid_dist(hands_dict[keyi][1],hands_dict[keyj][1])
            dist_dict[f'{keyi}'+f'{keyj}'] = list([distll,distlr,distrl,distrr])
 
    # calculating distances between hands and bodies
    for i,keyi in enumerate(keysh,start =1):
        for j,keyj in enumerate(keysb,start =1):
            if j>=i:
                break

            distlb = calc_euclid_dist(hands_dict[keyi][0],body_kp[keyj])
            
            distrb = calc_euclid_dist(body_kp[keyj],hands_dict[keyi][1])

            dist_dict[f'{keyi}'+f'{keyj}'].append(distlb)
            dist_dict[f'{keyi}'+f'{keyj}'].append(distrb)
                
    for i,keyi in enumerate(keysb,start =1):
        for j,keyj in enumerate(keysh,start =1):
            if j>=i:
                break

            distlb = calc_euclid_dist(hands_dict[keyj][0],body_kp[keyi])
            
            distrb = calc_euclid_dist(body_kp[keyi],hands_dict[keyj][1])
            

            dist_dict[f'{keyi}'+f'{keyj}'].append(distlb)
            dist_dict[f'{keyi}'+f'{keyj}'].append(distrb)


    # calculating distances between hands and heads
    for i,keyi in enumerate(keysh,start =1):
        for j,keyj in enumerate(keyhead,start =1):
            if j>=i:
                break

            distlh = calc_euclid_dist(hands_dict[keyi][0],head_kp[keyj])
            
            distrh = calc_euclid_dist(head_kp[keyj],hands_dict[keyi][1])
            
            dist_dict[f'{keyi}'+f'{keyj}'].append(distlh)
            dist_dict[f'{keyi}'+f'{keyj}'].append(distrh)



    for i,keyi in enumerate(keyhead,start =1):
        for j,keyj in enumerate(keysh,start =1):
            if j>=i:
                break

            distlh = calc_euclid_dist(hands_dict[keyj][0],head_kp[keyi])
            
            distrh = calc_euclid_dist(head_kp[keyi],hands_dict[keyj][1])
            
 
            dist_dict[f'{keyi}'+f'{keyj}'].append(distlh)
            dist_dict[f'{keyi}'+f'{keyj}'].append(distrh)



    # calculating distances between bodies
    for i,keyi in enumerate(keysb,start =1):
        for j,keyj in enumerate(keysb,start =1):
            if j>=i:
                break

            distbb = calc_euclid_dist(body_kp[keyi],body_kp[keyj])

            dist_dict[f'{keyi}'+f'{keyj}'].append(distbb)


    # calculating distances between heads
    for i,keyi in enumerate(keyhead,start =1):
        for j,keyj in enumerate(keyhead,start =1):
            if j>=i:
                break

            disthh = calc_euclid_dist(head_kp[keyi],head_kp[keyj])

            dist_dict[f'{keyi}'+f'{keyj}'].append(disthh)
    
    return dist_dict


def extract_hands_keypoints(results, threshold_class, threshold_keypoint):
    # creating a dictionary to collect keypoints to each object id as dictionary key
    existing_kp = {}
    for result,i_d in zip(results[0],results[0].boxes.id):
        # There results for bounding boxes, and confidence scores for general detect
        x1, y1, x2, y2,_, conf_for_detect, class_id_detected = (result.boxes.data.tolist())[0]
        # If the confidence score for general detect is lower than threshold, skip
        if conf_for_detect < threshold_class:
            continue
        # keypoints
        keys = (result.keypoints.data.tolist())[0]
        xl_key, yl_key, confl = keys[9]
        if confl > threshold_keypoint:
           l = [int(xl_key),int(yl_key)]
        else:
            l = []
        xr_key, yr_key, confr = keys[10]
        if confr > threshold_keypoint:
           r = [int(xr_key),int(yr_key)]
        else:
            r = []
        hands_coords = list([l,r])
        # Adding existing hand keypoints of an object in a frame to the dictionary   
        existing_kp[int(i_d)] = hands_coords
    return existing_kp

def extract_body_keypoints(results,threshold_class, threshold_keypoint):
    # creating a dictionary to collect keypoints to each object id as dictionary key
    existing_kp = {}
    for result,i_d in zip(results[0],results[0].boxes.id):
        # There results for bounding boxes, and confidence scores for general detect
        x1, y1, x2, y2,_, conf_for_detect, class_id_detected = (result.boxes.data.tolist())[0]
        # If the confidence score for general detect is lower than threshold, skip
        if conf_for_detect < threshold_class:
            continue
        # keypoints
        keys = (result.keypoints.data.tolist())[0]
        xl_key, yl_key, confl = keys[5]
        xr_key, yr_key, confr = keys[6]
        if (confl>threshold_keypoint) and (confr>threshold_keypoint):
            # Adding existing hand keypoints of an object in a frame to the dictionary   
            mid_point  = list([int((xr_key+xl_key)/2),int((yl_key+yr_key)/2)])
            
        else:
            mid_point = []

        existing_kp[int(i_d)] = mid_point

    return existing_kp

def extract_head_keypoints(results,threshold_class, threshold_keypoint):
    # creating a dictionary to collect keypoints to each object id as dictionary key
    existing_kp = {}
    for result,i_d in zip(results[0],results[0].boxes.id):
        # There results for bounding boxes, and confidence scores for general detect
        x1, y1, x2, y2,_, conf_for_detect, class_id_detected = (result.boxes.data.tolist())[0]
        # If the confidence score for general detect is lower than threshold, skip
        if conf_for_detect < threshold_class:
            continue
        # keypoints
        keys = (result.keypoints.data.tolist())[0]
        xh_key, yh_key, confh = keys[0]
        if confh>threshold_keypoint:
            # Adding existing hand keypoints of an object in a frame to the dictionary   
            mid_point  = list([int(xh_key),int(yh_key)])
        else:
            mid_point = []
        existing_kp[int(i_d)] = mid_point
    return existing_kp


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
        dist = int(sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1])))
        return dist
    else: 
        return np.nan
    

def preprocess_keypoints(keypoints):
    kepoint = np.array(keypoints)
    for i,line in enumerate(kepoint):
        nums = 0
        num_num = 0
        deviation = 0
        for num in line:
            if np.isnan(num) == False:        
                nums = nums + num
                num_num = num_num + 1

            if num_num == 0:
                line = np.nan_to_num(x = line,copy= False,nan = 0)
                continue

        mean = nums/num_num

        for num in line:
            if np.isnan(num) == False:        
                deviation = deviation + (num - mean)*(num-mean)

        std_dev = sqrt(deviation/num_num)

        for j,num in enumerate(line):
            if np.isnan(num) == False:  
                kepoint[i][j]= (num - mean)/std_dev

        line = np.nan_to_num(x = line,copy= False,nan = 0)
    return torch.Tensor([kepoint]).transpose(1,2)

#fight detection neural network loading
fight_net= torch.jit.load('fight_detection.pt')
fight_net.eval()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
modely = YOLO('yolov8l-pose.pt')  # load a pretrained YOLOv8n classification model
modely.to(device)

#Choose your video or stream there
video_path = r"D:\videos\hands3.mp4"
cap = cv2.VideoCapture(video_path)
# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS) # or number
# Create a VideoWriter object to save the output video
output_video_path = r"D:\videos_processed\fight1_processed.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


actions = ['fighting','not_fighting']
color_map = {'fighting': (0,100,200),'not_fighting': (200,100,0)}

font_scale = 1.6
thickness = 2

winsize = 40

distance_dict = {}

label_map = {num: label for num, label in enumerate(actions)}

ans = 'not_fighting'


while cap.isOpened():
# Read a frame from the video
    success, frame = cap.read()
    if success:
        time_yolo1 = time()
        results = modely.track(frame, persist=True, retina_masks=True, boxes=True, show_conf=False, line_width=1,  conf=0.6, iou=0.5,  classes=0, show_labels=False, device=device,verbose = False,tracker="bytetrack.yaml")
        time_yolo2 = time()
        yolo_time = time_yolo2 - time_yolo1 
        if results[0].boxes.id is not None:
            
            time_feature1 = time()
            #extracting keypoints
            body_kp = extract_body_keypoints(results = results,threshold_class=0.4,threshold_keypoint=0.4)
            hands_kp = extract_hands_keypoints(results = results,threshold_class=0.4,threshold_keypoint=0.4)
            head_kp = extract_head_keypoints(results = results,threshold_class=0.4,threshold_keypoint=0.4)
            #calculating distances between keypoints

            dd = calc_distances(hands_kp,body_kp,head_kp)
            time_feature2 = time()
            feature_time = time_feature2 - time_feature1
            #appending distances dictionary and evaluating average distance and classification based on it
            for key in dd.keys():

                if key not in distance_dict.keys():
                    distance_dict[key] = deque(maxlen=40)

                distance_dict[key].append(dd[key])
                
                if len(distance_dict[key]) == winsize:
                    time_preprocess1 = time()
                    keypoints = preprocess_keypoints(distance_dict[key])
                    time_preprocess2 = time()
                    preprocess_time = time_preprocess2- time_preprocess1
                    time_model1 = time()
                    logits = fight_net(keypoints.to(device,dtype = torch.float))
                    prediction = int(torch.argmax(logits, dim=1).cpu())
                    ans = label_map[prediction]
                    time_model2 = time()
                    model_time = time_model2 - time_model1
                    distance_dict[key].clear()
                    print(f"Time spent on YOLOv8 inference:{yolo_time:.3f}")
                    print(f"Time spent on keypoints extraction and feature extraction:{feature_time:.3f}")
                    print(f"Time spent on preprocessing features:{preprocess_time:.3f}")
                    print(f"Time spent on fight detection module inference:{model_time:.3f}")
                    if ans == 'fighting':
                        break
                    

            text_size, _ = cv2.getTextSize(ans, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_position = (frame_width - text_size[0] - 10, text_size[1] + 10)
            cv2.rectangle(frame, (text_position[0] - 5, text_position[1] - text_size[1] - 5),
                                    (text_position[0] + text_size[0] + 5, text_position[1] + 5), color=(0, 0, 0),
                                    thickness=cv2.FILLED)
            cv2.putText(frame, ans, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_map[ans], thickness, cv2.LINE_AA)


                    
            

        annotated_frame_show = cv2.resize(frame, (1080, 720))
        out.write(frame)
        cv2.imshow("YOLOv8 Inference", annotated_frame_show)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        
    else:
        # Break the loop if the end of the video is reached
        break


out.release()
cap.release()
cv2.destroyAllWindows()
