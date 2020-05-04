############################################
# Project: Pepper Jack Renderer
# File: renderer\gpu_renderer.py
# By: ProgrammingIncluded
# Website: ProgrammingIncluded.com
# Description:
#   Houses GPU rendering pipeline. Requires
# Numba and cudatoolkit installed.
############################################

# External
import math
import numpy as np
from numba import cuda

# PJR
from manifold import get_transform_matrix, create_manifold
from video_cache_loader import get_video_from_frame

THREADS_PER_BLOCK = 8

@cuda.jit
def cuda_render_loop(v_h, v_w, max_t, view_port, world_space, preloaded_images, loaded_t_values):
    x, y = cuda.grid(2)

    if x >= v_w or y >= v_h:
        return

    # Get the specific coordinates in world_space
    coord_column = x + (y * v_w)
    t = np.int32(world_space[2, coord_column])
    t_next = min(t + 1, max_t)

    # Get the arg location of the image
    img_lookup = 0
    for i in range(0, loaded_t_values.shape[0]):
        if loaded_t_values[i] == t:
            img_lookup = i
            break

    # Use the preloaded images to generate image
    # Use naive clipping algorithm for now
    wy = np.int32(world_space[1, coord_column])
    wx = np.int32(world_space[0, coord_column])

    # world coordinates can be in float
    # Use basic clipping for now
    # Clipping
    if wx < 0:
        wx = 0
    elif wx > v_w:
        wx = v_w

    if wy < 0:
        wy = 0
    elif wy > v_h:
        wy = v_h

    # Get the actual pixel value
    pixel_value = preloaded_images[img_lookup, wy, wx, :]

    # Use np.flip alternative.
    view_port[y, x, 0] = pixel_value[2]
    view_port[y, x, 1] = pixel_value[1]
    view_port[y, x, 2] = pixel_value[0]


def basic_render(video_metadata, cache_dir):
    # Calculate all the image frames necessary and preload them
    v_h = video_metadata["frame_height"]
    v_w = video_metadata["frame_width"]
    sample_delta = video_metadata["sample_delta"]
    max_t = video_metadata["total_frames"]

    # Camera coordinates
    camera_coord = np.arange(v_w * v_h).reshape((v_w, v_h))

    # Go from camera space to world space
    transformation_matrix = get_transform_matrix()
    camera_pixels = create_manifold(v_w, v_h)
    world_space = transformation_matrix @ camera_pixels

    # Go through each z axis and check the ranges
    z_axis_values = world_space[2, :].copy()
    world_space = world_space[0:3, :].astype(np.float32)
    z_axis_underround = z_axis_values.astype(np.int32)

    # Precreate the image manifold
    preloaded_images = []
    loaded_t_values = []
    t_values = np.unique(z_axis_underround)
    for t in t_values:
        t_next = min(t + 1, max_t)
        # Load the image before
        preloaded_images.append(get_video_from_frame(t, cache_dir, sample_delta))
        # Load the image after
        preloaded_images.append(get_video_from_frame(t_next, cache_dir, sample_delta))
        loaded_t_values += [t, t_next]

    # Convert to numpy for cuda to use
    preloaded_images = np.stack(preloaded_images).astype(np.int8)
    loaded_t_values = np.array(loaded_t_values).astype(np.int32)

    # Create view_port matrix
    view_port = np.empty((v_w, v_h, 3), dtype=np.int8)

    # Calculate blocks for CUDA
    num_blocks_x = math.ceil(v_w / THREADS_PER_BLOCK)
    num_blocks_y = math.ceil(v_h / THREADS_PER_BLOCK)

    # Launch the kernel!
    cuda_render_loop[(num_blocks_x, num_blocks_y), (THREADS_PER_BLOCK, THREADS_PER_BLOCK)](v_h, v_w, max_t, view_port, world_space, preloaded_images, loaded_t_values)
    return np.swapaxes(view_port, 0, 1)
