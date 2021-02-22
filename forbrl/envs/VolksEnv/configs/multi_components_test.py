# work dir
root_workdir = 'workdir'

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
        dict(type='CircleGridBoard', name='circle_board_1',
             shape=(7, 7), scale=0.015, corner=0.013, offset=0),
        dict(type='CircleGridBoard', name='circle_board_2',
             shape=(7, 7), scale=0.02, corner=0.01733, offset=0),
        dict(type='AprilTags', name='april_tags_1', family='hhh'),
        dict(type='AprilTagsSim', name='april_tags_sim_1', family='hhh'),
    ],
    vision_sensors=[
        dict(type='RealsenseCam', name='camera1',
             color_res=(1920, 1080), color_fr=15,
             depth_res=(1280, 720), depth_fr=15,
             serial_number='911222060451'),
        dict(type='RealsenseCam', name='camera2',
             color_res=(1280, 720), color_fr=15,
             depth_res=(1280, 720), depth_fr=15,
             serial_number='909522060990'),
    ]
)

# 3. cerebrum
cerebrum = dict(
    motor_cortex=[
        dict(type='PathSampler', name='path_sampler1',
             points=[
                 [-2.44121, -2.14992, 2.308032, -1.75258, -0.85728, -0.68924],
                 [-1.40368, -2.00700, 2.299051, -2.16302, -2.29230, 0.302270],
                 [-2.15230, -1.32297, 1.398167, -0.99853, -1.39997, -0.32017],
                 [-1.62747, -1.46954, 1.655560, -1.45825, -2.20364, -0.05106],
                 [-2.44012, -1.96320, 2.140986, -1.65201, -1.06973, -0.65055],
                 [-1.70195, -1.98520, 2.322428, -2.24354, -2.07213, -0.00107],
                 [-2.17122, -1.36631, 1.478761, -1.19337, -1.52531, -0.31793],
                 [-1.52811, -1.44959, 1.645168, -1.32840, -2.12390, 0.056704]
             ],
             max_path=100,
             path_step=4),
    ],
    posterior_parietal_cortex=[
        dict(type='AxyBSolver', name='axyb_solver1', self_check=True),
        dict(type='AxyBSolver', name='axyb_solver2', self_check=False),
    ],
    visual_cortex=[
        dict(type='CalHCam', name='cal_h_cam1',
             calib_board='circle_board_1', camera='camera1', thres=90,
             pnp_method='ransac'),
        dict(type='CalHCam', name='cal_h_cam2',
             calib_board='circle_board_1', camera='camera1', thres=90,
             pnp_method='ransac'),
    ]
)
