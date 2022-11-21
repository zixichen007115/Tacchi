import mitsuba
mitsuba.set_variant("scalar_rgb")
# mitsuba.set_variant("packet_rgb")

from mitsuba.core.xml import load_file
from mitsuba.core import Thread, LogLevel

logger = Thread.thread().logger()
logger.set_log_level(LogLevel.Warn)

from os import path as osp
from os.path import join

import imageio 
# to load exrs
imageio.plugins.freeimage.download()

from tactile_optical_simulation.scene_cfg_loading_utils import load_render_cfg, load_img_with_cfg, load_scene_from_params
from tactile_optical_simulation.folder_utils import create_folder

# Load file list 
from file_list_cfgs.flatgel import fname, img_list, project_name

cdir = osp.dirname(osp.abspath(__file__))

# Register any searchs path needed to load scene resources (optional)
dname = osp.dirname(fname)
Thread.thread().file_resolver().append(join(cdir, dname))

# load render params
render_params = load_render_cfg(join(cdir, "render_cfgs", "focussed.cfg"))
loading_param_list, target_im_list = load_img_with_cfg(img_list, render_params)

for scene_id, params in enumerate(loading_param_list):
  print(f"Rendering {scene_id}/{len(loading_param_list)} fn:{params['baseFn']}")

  # Load the scene from an XML file
  scene = load_file(fname, **render_params, **params)

  outFn = f"{params['baseFn']}_sim.exr"

  # create output dir
  create_folder(join("results", "flatgel"))
  outFn = join("results", "flatgel", outFn)
  
  scene.integrator().render(scene, scene.sensors()[0])

  # After rendering, the rendered data is stored in the film
  film = scene.sensors()[0].film()
  film.set_destination_file(outFn)
  film.develop()
