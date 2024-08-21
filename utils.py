import numpy as np
import os
import subprocess


def find_mp4_files_in_folders(folders):
    mp4_files = []
    for folder in folders:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.mp4'):
                    absolute_path = os.path.join(root, file)
                    relative_path = os.path.relpath(absolute_path, folder)
                    mp4_files.append((absolute_path, relative_path))
    return mp4_files

def find_relative_path(directory, filename):
    for root, dirs, files in os.walk(directory):
        if filename in files:
            absolute_path = os.path.join(root, filename)
            relative_path = os.path.relpath(absolute_path, directory)
            return relative_path
    return None

def find_mp4_files(folder):
    mp4_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.mp4'):
                mp4_files.append(os.path.join(root, file))
    return mp4_files


def convert_framerate(input_file, output_dir, output_name, framerate=30):
    # Combine output directory and output name to get the full path
    full_output_path = os.path.join(output_dir, output_name)
    
    # Get the directory part from the full output path
    output_directory = os.path.dirname(full_output_path)
    
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Split the input file into name and extension
    file_root, file_ext = os.path.splitext(full_output_path)
    
    # Create the output file name
    output_file = f"{file_root}_30fps{file_ext}"
    
    # Check if the output file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping conversion.")
        return output_file
    
    # Command to convert frame rate using FFmpeg
    command = [
        'ffmpeg',
        '-i', input_file,             # Input file
        '-r', str(framerate),         # Set frame rate
        output_file                   # Output file
    ]
    
    # Run the FFmpeg command
    subprocess.run(command, check=True)
    print(f"Converted video saved as: {output_file}")
    return output_file
def get_mp4_files(directories):
    mp4_files = []
    
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.mp4'):
                    mp4_files.append(os.path.join(root, file))
    
    return mp4_files

def linear_interpolate(points):
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
                std[i] = 1

        
        normalized_points = (points - mean) / std
        return normalized_points, mean, std

def denormalize(normalized_data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Denormalize a batched input array.
    
    Parameters:
    normalized_data (np.ndarray): The normalized data of shape [N, M, 2].
    mean (np.ndarray): The mean of the original data of shape [N, 2].
    std (np.ndarray): The standard deviation of the original data of shape [N, 2].
    
    Returns:
    np.ndarray: The denormalized data of shape [N, M, 2].
    """
    # Ensure mean and std are broadcastable to the shape of normalized_data
    mean = np.tile(mean,(normalized_data.shape[1],1)) # Reshape mean to [N, 1, 2]
    std = np.diag(std)    # Reshape std to [N, 1, 2]
    
    # print(mean.shape)
    # print(normalized_data[0].shape)
    # print(std.shape)
    # Denormalize
    denormalized_data = np.matmul(normalized_data[0],std) + mean
    return denormalized_data