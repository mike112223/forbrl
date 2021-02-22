# work dir
root_workdir = 'workdir'

# 1. logging
logger = dict(
    handlers=(
        dict(type='StreamHandler', level='INFO'),
    ),
)

# 2. equipment
equipment = dict(
    fiducial_markers=[
        dict(type='CircleGridBoard', name='circle_board_1',
             shape=(7, 7), scale=0.02, corner=0.018, offset=0.1),
    ],
    vision_sensors=[
        dict(type='RealsenseCam', name='camera1',
             color_res=(1920, 1080), color_fr=30,
             depth_res=(1280, 720), depth_fr=15,
             serial_number='938422076086'),  # 938422076086 911222060451
        #          preset='/media/yuhaoye/DATA7/git/Eye-in-Hand-calib/configs'
        #                 '/realsense_presets/realsense_set.json'),
    ]
)

# 3. cerebrum
cerebrum = dict(
    visual_cortex=[
        dict(type='CalHCam', name='cal_h_cam1',
             calib_board='circle_board_1', camera='camera1', thres=90,
             pnp_method='ransac'),
    ]
)
