##############################################
# Project: PepperJackRenderer
# File: video_cache_loader.py
# By: ProgrammingIncluded
# Website: ProgrammingIncluded.com
# Description:
#   Creates a cache of a video for processing.
###############################################

# Std
import os
import json
import hashlib

# External
import cv2
import numpy as np

# External function for easier numba support and serialization
def get_video_from_frame(frame, cache_dir, sample_delta):
    return cv2.imread(os.path.join(cache_dir, str(frame * (sample_delta)) + ".jpg"), cv2.IMREAD_COLOR).astype(np.uint8)

class VCL:
    """Uses CPU to preprocess video and saves them as jpg. Uses cv2 as backend."""
    def __init__(self):
        self.frame_width = 0
        self.frame_height = 0
        self.sample_interval_ms = 0
        self.raw_frame_count = 0
        self.fps = 0
        self.sample_delta = 0
        self.video_fpath = ""
        self.cache_dir = ""

    def get_frame(self, frame):
        return get_video_from_frame(frame, self.cache_dir, self.sample_delta)

    def get_total_frames(self):
        return int((self.raw_frame_count / self.sample_delta) + 1)

    def get_video_metadata(self):
        """Get metadata in form of dictionary for easier GPU support."""
        return {
            "frame_height": self.frame_height,
            "frame_width": self.frame_width,
            "sample_interval_ms": self.sample_interval_ms,
            "sample_delta": self.sample_delta,
            "fps": self.fps,
            "raw_frame_count": self.raw_frame_count,
            "total_frames": self.get_total_frames()
        }

    def process_video(self, video_fpath, cache_dir="video_cache", sample_interval_ms=100):
        """
            Processes video file and converts it into a series of jpgs
            inside a specified cache folder. Checks if cache has already been written
            if not, goes and writes to cache. Otherwise attempts to process the folder.
            Returns tuple of video stats: (self.frame_width, self.frame_height, self.raw_frame_count, self.fps)
        """
        cap = cv2.VideoCapture(video_fpath)

        # Assign video attributes
        self.video_fpath = video_fpath
        self.cache_dir = cache_dir
        self.sample_interval_ms = sample_interval_ms
        self.raw_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        # Other constants for later to calculate sampling
        sample_constant = int(round((self.fps / 1000) * sample_interval_ms))
        sample_constant = max(0, sample_constant)
        self.sample_delta = sample_constant

        # Video stats packaged as tuples for easier return later
        video_stats = (self.frame_width, self.frame_height, self.raw_frame_count, self.fps)

        # Hashing variables
        hash_values = [video_fpath, sample_interval_ms, self.raw_frame_count, self.frame_width, self.frame_height, self.fps]
        file_hash = hashlib.md5(" ".join([str(v) for v in hash_values]).encode("utf-8")).hexdigest()

        # Check if there is already a metadata file
        checksum_file = os.path.join(cache_dir, "check.json")
        if os.path.exists(checksum_file):
            with open(checksum_file) as f:
                try:
                    check = json.load(f)
                    if len(check) == 1 and check.get("sum") == file_hash:
                        cap.release()
                        print("Cache directory {} found. Using generated cache.".format(cache_dir))
                        return video_stats
                    else:
                        print("Invalid cache found with invalid hash. Regenerating cache folder.")
                except Exception as e:
                    print("Invalid checksum in cache. Clear cache folder first before continuing.")
                    exit(1)
        else:
            print("Generating cache folder.")

        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Negative one means sample for all frames
        if sample_interval_ms == -1:
            self.sample_delta = 1

        for idx in range(0, self.raw_frame_count):
            _, img = cap.read()

            if idx % self.sample_delta != 0:
                continue

            cv2.imwrite(os.path.join(cache_dir, "{}.jpg".format(idx)), img)

        # Release video resources
        cap.release()

        # Write checksum file
        with open(checksum_file, "w+") as f:
            json.dump({"sum": file_hash}, f)

        return video_stats