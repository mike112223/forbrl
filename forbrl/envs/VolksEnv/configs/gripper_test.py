# work dir
root_workdir = 'workdir'

# 1. logging
logger = dict(
    handlers=(
        dict(type='StreamHandler', level='DEBUG'),
    ),
)

# 2. equipment
equipment = dict(
    end_effectors=[
        dict(type='InspireGripper', name='gripper_1'),
    ],
)
