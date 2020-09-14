import numpy as np
import xml.etree.ElementTree as ET

from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from utils import MicrowaveObject
from robosuite.utils.mjcf_utils import array_to_string

# World setup
world = MujocoWorldBase()

'''
mujoco_robot = Panda()
gripper = gripper_factory('PandaGripper')
gripper.hide_visualization()
mujoco_robot.add_gripper(gripper)
mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)
'''
mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0., 0.])
world.merge(mujoco_arena)

microwave = MicrowaveObject()

for a in microwave.get_assets():
    world.asset.append(a)

mw_body = microwave.get_visual(tag_name='cabinet_bottom')
table_top_pos = mujoco_arena.table_top_abs
mw_body.set('pos', array_to_string(table_top_pos))
mw_body.set('quat', '0. 0. 0. 1.')
world.worldbody.append(mw_body)

act = microwave.get_actuator(jnt_name="bottom_left_hinge")
world.actuator.append(act)

model = world.get_model(mode="mujoco_py")

# Simulation
from mujoco_py import MjSim, MjViewer

sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0
print(len(sim.data.ctrl))

#import pdb; pdb.set_trace()

for i in range(2000):
    #sim.data.ctrl[:] = 0
    sim.data.ctrl[-1] = 0.01
    sim.step()
    viewer.render()

input('should try to close it now?')

for i in range(2000):
    #sim.data.ctrl[:] = 0
    sim.data.ctrl[-1] = -0.01
    sim.step()
    viewer.render()



