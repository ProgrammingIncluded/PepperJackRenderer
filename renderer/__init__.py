############################################
# Project: Pepper Jack Renderer
# File: renderer\__init__.py
# By: ProgrammingIncluded
# Website: ProgrammingIncluded.com
############################################

def basic_render(video_metadata, cache_dir, cpu_only):
    """Rendering interface function that can select between gpu and cpu rendering"""
    if not cpu_only:
        try:
            import numba
            import renderer.gpu_renderer as gpur
            return gpur.basic_render(video_metadata, cache_dir)
        except Exception as e:
            print("Unable to use GPU renderer. Defaulting to CPU. {}".format(e))

    import renderer.cpu_renderer as cpur
    return cpur.basic_render(video_metadata, cache_dir)

__all__ = [basic_render]