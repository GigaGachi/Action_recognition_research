import numpy as np
import cv2 as cv
from traj_predictor import traj_extractor,extractor_and_predictor,ray_tune,initialize_model_from_pt,train_best_model, npy_processor, sequence_dataset, trajectory_predictor
from yolo_traj import yolo_traj_extractor
import os
import subprocess
from ray import train, tune
import torch
from utils import convert_framerate, find_mp4_files_in_folders

# "/home/iustimov/datasets/drone-tracking-datasets/"

video_dirs = ["/mnt/nas/ue/UE_STUFF/Photos/Moving Target/DRONE_RGB_DARK/",
              "/mnt/nas/ue/UE_STUFF/Photos/Moving Target/DRONE_RGB_DARKER_FOG/",
              "/mnt/nas/ue/UE_STUFF/Photos/Moving Target/DRONE_VISIBLE/"]



test_video  = "/home/iustimov/Videos/trajectory/traj_3.mp4"

output_dir = "/home/iustimov/Videos/trajectory/30_fps/synthetic"

video_paths  = find_mp4_files_in_folders(video_dirs)

print(video_paths)

for video_path in video_paths:
    # model = traj_extractor()
    # model.debug = False
    video_path = convert_framerate(video_path[0],output_dir,video_path[1], 30)
    # if (video_path == -1):
    #     continue
    # model.test_create_trajectories_sparse(video_path)
    
    
converted_videos_dirs = ["/home/iustimov/Videos/trajectory/30_fps/synthetic"]

converted_videos_paths = find_mp4_files_in_folders(converted_videos_dirs)

traj_path = "/home/iustimov/tasks/traj_pred/trajectories_onlydrone"

for video_path in video_paths:
    model = traj_extractor()
    model.debug = False
    model.test_create_trajectories_sparse(video_path[0])
    
npy_loader = npy_processor(traj_path)

list_of_sequences = npy_loader.process_dataset(traj_path)

seq_data= sequence_dataset(list_of_sequences,15,15)

print(len(seq_data))
arrays = npy_loader.process_dataset(npy_loader.path_to_npy_directory)
config_hp = {
    "learning_rate": tune.loguniform(0.00001,0.01),
    "batch_size": tune.choice([256,512.1024]),
    "hidden_size": tune.randint(2,64),
    "num_layers": tune.randint(2,32),
    "input_seq_len": 15,
    "output_seq_len": 15,
    "dropout": tune.uniform(0.0, 0.3),
}
    
res = ray_tune(config_hp=config_hp,num_samples=1000)

# Best_config = {'learning_rate': 0.0058655592208453665, 'batch_size': 512, 'hidden_size': 34, 'num_layers': 2, 'input_seq_len': 15, 'output_seq_len': 15, 'dropout': 0.09163596681405999}

# best_model = train_best_model(Best_config,True)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# example_input = torch.randn((1, Best_config["input_seq_len"], 2)).to(device=device)
# traced_model = torch.jit.trace(best_model,example_input)
# traced_model_path = "traced_trajectory_predictor.pt"
# traced_model.save(traced_model_path)

# best_model = initialize_model_from_pt(Best_config,"traced_trajectory_predictor.pt")

# traj_extr = traj_extractor()

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# holy_moly = extractor_and_predictor(model=best_model,traj_extractor=traj_extr,debug=True,device=device)

# holy_moly.inference_on_video(test_video)