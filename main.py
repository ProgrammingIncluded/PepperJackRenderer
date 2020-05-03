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

# PJR
from video_cache_loader import VCL

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
    parser.add_argument("--cpu-only", "-c", action="store_true", default=False, help="Render using cpu only.")
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
        from renderer import basic_render
        img = basic_render(video_metadata, video.cache_dir, cpu_only=args.cpu_only)
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