##############################################
# Project: PepperJackRenderer
# File: main.py
# By: ProgrammingIncluded
# Website: ProgrammingIncluded.com
# Description:
#    Take a video and slice it through time.
# Inspired by CGMatter's video on Blender.
###############################################

# Std
import os
import argparse

# Externals
import numpy as np
import matplotlib.pyplot as plt

# TimeSlice
from video_cache_loader import VCL, get_video_from_frame
from manifold import get_transform_matrix, create_manifold


def basic_render(video_metadata, cache_dir):
    """Basic render loop where we assume the manifold is a plane intersection."""
    v_h = video_metadata["frame_height"]
    v_w = video_metadata["frame_width"]

    # Camera coordinates
    camera_coord = np.arange(v_w * v_h).reshape((v_w, v_h))

    # Go from camera space to world space
    transformation_matrix = get_transform_matrix()

    # Transform camera pix coordinates to color space coordinates
    camera_pixels = create_manifold(1280, 720)
    world_space = transformation_matrix @ camera_pixels
    print("World space dimensions {}".format(world_space.shape))


    # Go through each z axis and check the ranges
    z_axis_values = world_space[2, :]
    z_axis_underround = z_axis_values.astype(int)
    x_y_z_world_space = world_space[0:3, :]

    # Get all unique values
    t_values = np.unique(z_axis_underround)
    print("Frames hit: {}".format(t_values))

    view_port = np.empty((v_w, v_h, 3),dtype=np.int32)
    sample_delta = video_metadata["sample_delta"]
    max_t = video_metadata["total_frames"]

    # Accelerate loop using numba
    count = 0
    for t in t_values:
        print("Rendering: {}%".format(count/len(t_values)))
        count += 1

        # Between frames
        t_next = min(t + 1, max_t)

        img_data_t = None
        img_data_tp = None

        # Calculate the collision of points
        result = np.logical_and((x_y_z_world_space[2] >= t),(x_y_z_world_space[2] <= t_next))
        for col in np.argwhere(result):
            if img_data_t is None:
                img_data_t = get_video_from_frame(t, cache_dir, sample_delta)
                img_data_tp = get_video_from_frame(t_next, cache_dir, sample_delta)
        
            # Calculate the pixel value to get
            py, px = x_y_z_world_space[1, col], x_y_z_world_space[0, col]
            # Clip the values
            px = np.clip(int(px), 0, v_w)
            py = np.clip(int(py), 0, v_h)
        
            # Map the pixel color space to the result
            # Assume t for now
            # TODO: use z axis to calculate interpolation
            if img_data_t is None:
                print(img_data_t)
                print(t)
            pixel_value = img_data_t[py, px, :]
        
            # Use lookup table to get the pixel transformation mapping
            # Get the image data and assign to pixel
            view_port[camera_pixels[0, col], camera_pixels[1, col], :] = np.flip(pixel_value)
    return np.swapaxes(view_port, 0, 1)


def clear_output_cache_file(output_cache_path):
    """Clears output cache file if applicable"""
    for filename in os.listdir(output_cache_path):
        if filename.endswith(".jpg") or filename.endswith(".json"):
            os.remove(os.path.join(output_cache_path, filename))

    os.removedirs(output_cache_path)

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Take an mp4 video and slide it by given dimension, positional vector, and delta.")
    parser.add_argument("--video-path", "-v", type=str, help="File path to video file.")
    parser.add_argument("--input-cache-path", "-i", type=str, default="video_cache", help="File path to input cache file.")
    parser.add_argument("--sample-interval-ms", "-s", type=int, default=100, help="Only analyze every n ms of the video to save cache size. Supply -1 to create cache and analyze every frame.")
    parser.add_argument("--output-cache-path", "-o", type=str, default="output_cache", help="File path to output cache file.")
    args = parser.parse_args()

    # Check if the output cache path exists and empty if not, throw an error
    if not os.path.exists(args.output_cache_path):
        print("Creating output path folder: {}".format(args.output_cache_path))
        os.makedirs(args.output_cache_path)
    elif len(os.listdir(args.output_cache_path)) != 0:
        print("Output cache must be empty.")
        exit(1)

    # Process video in to sizable cache
    video = VCL()
    video.process_video(args.video_path, cache_dir=args.input_cache_path, sample_interval_ms=args.sample_interval_ms)
    video_metadata = video.get_video_metadata()
    print("Sample Frame Intervals: {}".format(video_metadata["sample_delta"]))

    # Render Loop
    try:
        img = basic_render(video_metadata, video.cache_dir)
        plt.imshow(img)
        plt.show()
    except Exception as e:
        print("Removing {}".format(args.output_cache_path))
        print("An error has occurred while rendering: {}".format(e))
        clear_output_cache_file(args.output_cache_path)
        exit(1)

    # Cleanup app
    clear_output_cache_file(args.output_cache_path)

if __name__ == '__main__':
    main()