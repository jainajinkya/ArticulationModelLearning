import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder

from robosuite.models.objects import MujocoGeneratedObject
from robosuite.utils.mjcf_utils import new_body, new_geom, new_site
from robosuite.utils.mjcf_utils import RED, GREEN, BLUE

import xml.etree.ElementTree as ET


# Arrow object
class ArrowObject(MujocoGeneratedObject):
    def __index__(self, pos=None, size=None, rgba=RED, group=1):
        super().__init__()
        if size is None:
            size = [0.02, 0.02, 0.2]
        if pos is None:
            pos = [0., 0., 0.]

        self.arrow = new_body()
        self.arrow.append(new_geom(geom_type="box", size=size, pos=pos, rgba=rgba, group=group))
        #mjGEOM_ARROW


def visualize_prediction(scene_filename, gt_axes, pred_axes):
    tree = ET.parse(scene_filename)
    root = tree.getroot()


    # Load scene in mujoco
    model = load_model_from_path(scene_filename)
    sim = MjSim(model)
    modder = TextureModder(sim)
    # viewer=MjViewer(sim)

    # Plot GT axis and predicted axis
    # Take snapshot

    print("Should have saved a snapshot")