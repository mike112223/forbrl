
root_workdir = 'logs'
gpu_id = '0'
num_gpu = len(gpu_id.split(','))

seed = 1234
deterministic = True

agents = dict(
    type='VPGAgent',
    algorithm=dict(
        type='DQN',
        model=dict(
            type='VPGNet',
            backbone=dict(
                type='ResNet',
                arch='resnet18',
                pretrained=True,
                frozen_stages=-1,
                in_dim=3,
                norm_eval=False
            ),
            head=dict(
                type='FCN',
                in_channels=[1024, 64],
                out_channels=[64, 1],
                norm_cfg=dict(type='BN'),
            ),
            mean=[0.485, 0.456, 0.406, 0.01, 0.01, 0.01],
            std=[0.229, 0.224, 0.225, 0.03, 0.03, 0.03],
            num_rotations=16,
            size_divisor=32

        ),
        criterion=dict(
            type='SmoothL1Loss',
            reduce=False,
        ),
        optimizer=dict(
            type='SGD',
            lr=1e-4,
            momentum=0.9,
            weight_decay=2e-5
        ),
        gamma=0.5
    ),
    memory=dict(
        type='VPGReplay',
        batch_size=2,
        max_size=5000,
        use_cer=True,
    ),
    policy=dict(
        type='VPGPolicy',
    ),
    base_explore=0.5,
    min_explore=0.1
)

envs = dict(
    type='VPGEnv',
    env='configs/vpg/vpg_game.py'
)

runner = dict(
    type='Runner',
    max_iter=2500,
)
