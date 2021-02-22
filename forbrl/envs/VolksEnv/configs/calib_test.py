# work dir
root_workdir = '/media/data/Tianhe/workspace/EyeInHandCalibrate'

# 1. logging
logger = dict(
    handlers=(
        dict(type='StreamHandler', level='INFO'),
        dict(type='FileHandler', level='DEBUG'),
    ),
)

# 2. equipment
equipment = dict(
    fiducial_markers=[
        dict(type='CircleGridBoard', name='circle_board1',
             shape=(7, 7), scale=0.02, corner=0.018, offset=0.1),
    ],
    robotic_arms=[
        dict(type='URArm', name='ur_arm1', host='192.168.1.169')
    ],
    vision_sensors=[
        dict(type='RealsenseCam', name='camera1',
             color_res=(1280, 720), color_fr=30,
             depth_res=(1280, 720), depth_fr=15,
             serial_number='938422076086'),
    ],
    end_effectors=[
        dict(type='InspireGripper', name='gripper_1'),
    ],
)

# 4. runner
runner = dict(
    type='CalibVerifier',
    # settings for environment
    arm='ur_arm1', cam='camera1', calib_board='circle_board1',
    gripper='gripper_1',
    # settings for runner
    acc=2, vel=2, step=0.01, verify_by_cam=False,
    session_dir=(root_workdir + '/Session_2020-06-15T18-46-18/results'),
)
