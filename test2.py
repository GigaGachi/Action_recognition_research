import numpy as np
import cv2 as cv
from traj_predictor import traj_extractor,extractor_and_predictor,ray_tune_qat,train_quantized_model, ray_tune,initialize_model_from_pt,train_best_model, npy_processor, sequence_dataset, trajectory_predictor
from yolo_traj import yolo_traj_extractor
import os
import subprocess
from ray import train, tune
import torch
from utils import convert_framerate, find_mp4_files_in_folders
import json
from torch.fx import symbolic_trace


def raytune_func(traj_path,quantized):

    config_hp = {
        "learning_rate": tune.loguniform(0.00001,0.1),
        "batch_size": tune.choice([2048,4096,8192,16384]),
        "hidden_size": tune.randint(2,64),
        "num_layers": tune.randint(2,32),
        "input_seq_len": tune.randint(10,20),
        "output_seq_len": tune.randint(15,20),
        "dropout": tune.uniform(0.0, 0.3),
        'traj_path': traj_path,
        "ams_grad": tune.choice([True,False]),
        "weight_decay": tune.uniform(0.001, 0.3),
    }
    
    if (quantized):
        res = ray_tune_qat(config_hp=config_hp,num_samples=2)
    else:
        res = ray_tune(config_hp=config_hp,num_samples=400)
    return res

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

def save_onnx(model,config,path):
    device = torch.device('cpu')
    model.to(device)
    example_input = torch.randn((1, config["input_seq_len"], 2)).to(device=device)

    torch.onnx.export(model, 
                  example_input, 
                  path, 
                  export_params=True, 
                  opset_version=19, 
                  do_constant_folding=True, 
                  input_names=['input'], 
                  output_names=['output']
                  )
    
def save_pt(model,config,path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    example_input = torch.randn((1, config["input_seq_len"], 2)).to(device=device)
    model.to(device)
    traced_model = torch.jit.trace(model,example_input)
    traced_model.save(path)

def save_pth(quantized_model,config,path):


    # print(quantized_model)
    
    torch.save(quantized_model.state_dict(), path)


video_dirs = ["/mnt/nas/ue/UE_STUFF/Photos/Moving Target/DRONE_RGB_DARK/",
              "/mnt/nas/ue/UE_STUFF/Photos/Moving Target/DRONE_RGB_DARKER_FOG/",
              "/mnt/nas/ue/UE_STUFF/Photos/Moving Target/DRONE_VISIBLE/"]

converted_videos_dirs = ["/home/iustimov/Videos/trajectory/30_fps/synthetic"]

traj_path = "/home/iustimov/tasks/traj_pred/trajectories_onlydrone"

test_video  = "/home/iustimov/Videos/trajectory/traj_3.mp4"

output_dir = "/home/iustimov/Videos/trajectory/30_fps/synthetic"

traced_model_path = "traced_trajectory_predictor_only_synthetic_quant.pt"

onnx_path = "models/traj_pred.onnx"

onnx_path_qat = "models/traj_pred_qat.onnx"

quantized_pth_path = "models/traj_pred_qat.pth"
# convert_videos(video_dirs,output_dir)
    
# extract_trajectories(converted_videos_dirs)

#Best_config_quant = raytune_func(traj_path,True)

Best_config = {'learning_rate': 0.023601244309060986, 'batch_size': 16384, 'hidden_size': 63, 'num_layers': 2, 'input_seq_len': 12, 'output_seq_len': 12, 'dropout': 0.17090514543892257, 'traj_path': '/home/iustimov/tasks/traj_pred/trajectories_onlydrone', 'ams_grad': False, 'weight_decay': 0.019441578482764302}
Best_config_quant = {'learning_rate': 0.023601244309060986, 'batch_size': 16384, 'hidden_size': 63, 'num_layers': 2, 'input_seq_len': 12, 'output_seq_len': 12, 'dropout': 0.17090514543892257, 'traj_path': '/home/iustimov/tasks/traj_pred/trajectories_onlydrone', 'ams_grad': False, 'weight_decay': 0.019441578482764302}

# with open('onlydrone_hp_config_quant.json', 'w') as f:
#     json.dump(Best_config, f)

model = best_model = train_quantized_model(Best_config,True)

save_pth(model,Best_config,quantized_pth_path)