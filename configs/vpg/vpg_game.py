
import numpy as np


root_workdir = 'logs'

workspace = np.asarray(
    [[-0.724, -0.276],
     [-0.224, 0.224],
     [-0.0001, 0.4]])
mode = 'blocking'

# 1. logging
logger = dict(
    handlers=(
        dict(type='StreamHandler', level='INFO'),
        dict(type='FileHandler', level='DEBUG'),
    ),
)

# 2. equipment
equipment = dict(
    end_effectors=[
        dict(type='InspireGripperSim',
             name='gripper1',
             handle_name='RG2_openCloseJoint',
             tcp=np.array([0, 0, 0.026]),
             speed=0.5,
             power=100,
             openmax=0.0536,
             openmin=-0.047,
             time_delay=0.5,
             mode=mode),
    ],
    objects=[
        dict(type='Primitive',
             name='obj1',
             sev_name='remoteApiCommandServer',
             func_name='importShape',
             num_obj=10,
             obj_mesh_dir='/home/yj/media_smart/github/forbrl/forbrl/envs/'
                          'VolksEnv/environment/equipment/objects/primitives/blocks',
             workspace=workspace,
             drop_height=0.15,
             drop_offset=0.1,
             color_space=np.array(
                 [[78.0, 121.0, 167.0],
                  [89.0, 161.0, 79.0],
                  [156, 117, 95],
                  [242, 142, 43],
                  [237.0, 201.0, 72.0],
                  [186, 176, 172],
                  [255.0, 87.0, 89.0],
                  [176, 122, 161],
                  [118, 183, 178],
                  [255, 157, 167]]) / 255.0,
             mode=mode),
    ],
    robotic_arms=[
        dict(type='URArmSim',
             name='arm1',
             handle_name='UR5_target',
             mode=mode)
    ],
    sim_environments=[
        dict(type='Vrep',
             name='sim1',
             address='127.0.0.1',
             port=19997,
             mode=mode)
    ],
    vision_sensors=[
        dict(type='RealsenseCamSim',
             name='cam1',
             handle_name='Vision_sensor_persp',
             color_res=(640, 480),
             presp_angle=54.7,
             clipping=(0.01, 10),
             depth=1.,
             mode=mode),
    ],
)

# 3. cerebrum
# 4. runner
runner = dict(
    type='VPG',
    sim='sim1', arm='arm1', camera='cam1',
    gripper='gripper1', obj='obj1', work_dir=root_workdir,
    workspace=workspace, resolution=0.002,
    num_rotations=16,
    grasp_reward=1., push_reward=0.5,
    grasp_loc_margin=0.15, push_margin=0.1,
    push_length=0.1,
    pixel_thresh=300, depth_thresh=[0.01, 0.3],
    no_change_thresh=10, empty_threshold=300
)
