import numpy as np
import cv2 as cv
from traj_predictor import traj_extractor,extractor_and_predictor,ray_tune,initialize_model_from_pt,train_best_model, npy_processor, sequence_dataset, trajectory_predictor
from yolo_traj import yolo_traj_extractor
import os
import subprocess
from ray import train, tune
import torch
from utils import convert_framerate, find_mp4_files_in_folders

def raytune_func(traj_path):

    config_hp = {
        "learning_rate": tune.loguniform(0.00001,0.1),
        "batch_size": tune.choice([256,512,1024,2048,4096]),
        "hidden_size": tune.randint(2,64),
        "num_layers": tune.randint(2,32),
        "input_seq_len": 15,
        "output_seq_len": 15,
        "dropout": tune.uniform(0.0, 0.3),
        'traj_path': traj_path
    }
        
    res = ray_tune(config_hp=config_hp,num_samples=200)
    
def extract_trajectories(converted_videos_dirs):

    converted_videos_paths = find_mp4_files_in_folders(converted_videos_dirs)

    for video_path in converted_videos_paths:
        print(video_path)
        model = traj_extractor()
        model.debug = False
        model.test_create_trajectories_sparse(video_path[0])
        
def convert_videos(video_dirs,output_dir):
    video_paths  = find_mp4_files_in_folders(video_dirs)
    for video_path in video_paths:
        video_path = convert_framerate(video_path[0],output_dir,video_path[1], 30)

def train_best_model_sequence(Best_config,traced_model_path):
    # Best_config = {'learning_rate': 0.0058655592208453665, 'batch_size': 512, 'hidden_size': 34, 'num_layers': 2, 'input_seq_len': 15, 'output_seq_len': 15, 'dropout': 0.09163596681405999}
    # Best_config_2 = {'traj_path': "/home/iustimov/tasks/traj_pred/trajectories_onlydrone",'learning_rate': 0.0025493047767871295, 'batch_size': 256, 'hidden_size': 54, 'num_layers': 4, 'input_seq_len': 15, 'output_seq_len': 15, 'dropout': 0.0849511364102184}
    best_model = train_best_model(Best_config,True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    example_input = torch.randn((1, Best_config["input_seq_len"], 2)).to(device=device)
    traced_model = torch.jit.trace(best_model,example_input)
    traced_model.save(traced_model_path)

video_dirs = ["/mnt/nas/ue/UE_STUFF/Photos/Moving Target/DRONE_RGB_DARK/",
              "/mnt/nas/ue/UE_STUFF/Photos/Moving Target/DRONE_RGB_DARKER_FOG/",
              "/mnt/nas/ue/UE_STUFF/Photos/Moving Target/DRONE_VISIBLE/"]

converted_videos_dirs = ["/home/iustimov/Videos/trajectory/30_fps/synthetic"]

traj_path = "/home/iustimov/tasks/traj_pred/trajectories_onlydrone"

test_video  = "/home/iustimov/Videos/trajectory/traj_3.mp4"

output_dir = "/home/iustimov/Videos/trajectory/30_fps/synthetic"

traced_model_path = "traced_trajectory_predictor_only_synthetic.pt"

# convert_videos(video_dirs,output_dir)
    
# extract_trajectories(converted_videos_dirs)

Best_config = raytune_func(traj_path)

train_best_model_sequence(Best_config=Best_config, traced_model_path= traced_model_path)

