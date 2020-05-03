# Pepper Jack Renderer

What happens if you were somehow able to observe history from a perspective outside of time?
Pepper Jack Renderer (PJR) is a 2D video renderer that renders videos 3-dimensionally by taking a video file and taking cross section chunks out of the video.
PJR does this by mapping the video space as x and y axis of a virtual world and takes the z axis a the time.
Inside this space, the video and its contents are a rectangular prism.
PJR then projects a viewport manifold that slices through the prism and displays the contents as an image.

PJR uses `Numba` and `CUDA` to speed-up rendering using the `gpu`. However, users can opt to use the `cpu` only mode.
Special thanks to CGMatter who had inspired this program in the video titled [video is 3D. not 2D](https://www.youtube.com/watch?v=NZFxQXe7LMM).

# How to Run

- Install dependencies. If using CPU-only mode use:

> conda install --file cpu-requirements.txt

Otherwise:

> conda install --file gpu-requirements.txt

- Run the Python command:

> python main.py -v video_file_path

`video_file_path` describes the path to the video. The binary will create a `video_cache` and `output_cache` folder which will contain `jpg` stillshots of the video and final render respectively.
Please make sure to have the proper permissions before starting the program.
If the file is large, try increasing the `sample_interval_ms` first in order do a basic sanity check. It reduces the amount of frames generated from the video and may reduce the time it takes to render. See `--help` for more options.

# How to Install
As also listed in `gpu-requirements.txt`:

* Python 3.x+
* cv2
* Numba (optional)
* CUDA (optional)

We recommend using `conda` distribution of `python3`. `xxx-requirements.txt` is written using packages found in `conda install` environment.
