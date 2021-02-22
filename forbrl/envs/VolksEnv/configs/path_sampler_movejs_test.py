# work dir
root_workdir = '/media/data/Tianhe/workspace/EyeInHandCalibrate'

# 1. logging
logger = dict(
    handlers=(
        dict(type='StreamHandler', level='INFO'),
    ),
)

# 2. equipment
equipment = dict(
    robotic_arms=[
        dict(type='URArm', name='ur_arm1', host='192.168.1.169'),
    ],
    vision_sensors=[
        dict(type='RealsenseCam', name='camera1',
             color_res=(1280, 720), color_fr=30,
             depth_res=(1280, 720), depth_fr=15,
             serial_number='938422076086'),
    ],
)

# 3. cerebrum
cerebrum = dict(
    motor_cortex=[
        dict(type='PathSampler', name='path_sampler1',
             points=[
                 [-2.85619, -1.90040, 2.03548, -1.56648, -1.15323, -1.10340],
                 [-2.05362, -2.21485, 2.36328, -2.22952, -1.96335, -0.43043],
                 [-2.66179, -1.67660, 1.84298, -1.50561, -1.18273, -1.10888],
                 [-2.11855, -1.78111, 2.00976, -1.92297, -1.92877, -0.57225],
                 [-2.70682, -1.81897, 1.96270, -1.49758, -1.22977, -1.11680],
                 [-2.20970, -2.15816, 2.34026, -2.21193, -1.84978, -0.40235],
                 [-2.71750, -1.65614, 1.81270, -1.30638, -1.23668, -1.16055],
                 [-2.02844, -1.72941, 2.02518, -2.02008, -2.14826, -0.56727]
             ],
             max_path=100,
             path_step=5),
    ],
)

# 4. runner
runner = dict(
    type='PathVerifier',
    arm='ur_arm1', cam='camera1', path_generator='path_sampler1',  # env setting
    delay=2, acc=2, vel=2, radius=0.01,   # runner setting
)
