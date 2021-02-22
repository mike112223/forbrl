# VolksEnv
An interface for manipulating a robotic-related environment both in reality
 and simulation. 

## Intro
Volks Environment creates an interface to manipulate equipment both in a real
 or simulated scenario. Two main components in VolksEnv: `Equipment` and
  `Cerebrum`.

- `Equipment` contains control interface for equipment real/sim cases. 

- `Cerebrum`, like its name, works as a 'brain' for the environment. Similar
 to the human brain, `Cerebrum` have different 'cortices' that contains a
  different type of functionality. (e.g an _AprilTags detector_ would be in
   the `Visual Cortex`, a _path planner_ would be in the `Motion Cortex`)
  
VolksEnv provides different levels of usage(you can see detail [here](#usages
)) that makes it flexible for you to use: you can use equipment only, you
 can make use of functional in cortices in the cerebrum, you can also create
  your own runner for more complex tasks.
   
### A word for reviewer
The environment is not a finished project yet. 
For now, the simulation part is still in progress. Here is a list of files
 that you can skip first during reviewing and corresponding reason(you can
  come back and review them after you finish the rest if you like. But 90% of
   them are not my code):
  
| **Files**                                                     | **Excuses**                                                                       |
|---------------------------------------------------------------|-----------------------------------------------------------------------------------|
| environment/equipment/×_sim                                   | Simulation part is still in progress                                              |
| environment/equipment/end_effectors/grippers/inspire.py       | Only slightly modified from original code provided by Inspire                     |
| environment/equipment/robotic_arms/urx                        | Only added some features such as auto reconnect, fancy movel, fancy movels        |
| environment/utils                                             | Most of files in utils are borrowed from volksseg and potentially from mmdetection |
| environment/cerebrum/posterior_parietal_cortex/axyb_solver.py | Most of codes in this file are borrowed from another repo                         |


Also, there are some TODOs in the code as you can see. Those are
 modifications I intended to make, you are welcomed to leave comments on
  those as well.

## Getting Started

These instructions will get you a copy of the project up and running on your
 local machine for development and testing purposes.

### Prerequisites & Installing

1. Clone this repo
2. Make sure you have python >= 3.6 installed
3. Install packages listed in the [requirements.txt](requirements.txt)
    - numpy~=1.16.3
    - six~=1.12.0
    - addict~=2.2.1
    - tqdm~=4.44.1
    - pyrealsense2~=2.27.0.1067
    - math3d~=3.3.4
    - pyserial~=3.4
    - opencv-python~=4.1.0
    
for example:

```
conda create -n volksenv python=3.7
conda activate volksenv
pip install -r requirements.txt
```

### Usages

There are different ways to use VolksEnv for your project. The main workflow
 is to:
1. Define your task. 
2. Decided to what level you would like to use the VolksEnv.
3. Create/modify files in [configs](configs) or [tasks](tasks) or even
 [runners](environment/runners) and more.

To run the following examples, you need to have access to the corresponding
  equipment and just:
```
cd volksenv
python tasks/_example_name.py_ --config _corresponding_config_file_
```
Here are examples in different levels:
 
#### Equipment level

This is for the scenario when you just want to manipulate equipment in simple
 logic. At equipment level, you only want to control the equipment and don't
  even need cerebrum to help you.
  
For example, in the case you only want to move the gripper in your own
 project. You can refer to [gripper_test](tasks/gripper_test.py). It's
  [config](configs/gripper_test.py) is fairly simple. You can start from
   there and integrate VolksEnv in your project.
   
#### Environment level

As your project goes more complex, you might want to connect your equipment
 using `Cerebrum`. At this level, you are making use of some more complex
  functional for your project.
 
For example, in [offset_test](tasks/offset_test.py) you want to get the
 spacial relation between your camera and a calibration board: the distance
  between their local coordinate systems' origin, the distance of camera to
   the surface of the board or even more complex information. As you can see in
    the [config](configs/offset_test.py), you need `Visual Cortex` to
     actually see the board, and inference the translation by what camera see.
    
#### Runner level

As your project goes even more complex: starting from the last example where
 you can get offset of a camera to the calibration board, you now want to use
  that to do an eye-in-hand calibration, You will find that the main function
   gets more and more complex and undesirably long. For a more complex task
   , you can create a runner to be more efficient.

See [calib_eye_in_hand](tasks/calib_eye_in_hand.py) and it's [config](configs/calib_eye_in_hand.py), you may notice that aside from
 `equipment` and `cerebrum`, there is a `runner` in the config as well. This
  [runner](environment/runners/eye_in_hand_calibrator.py) describes the
   process of calibration using equipment and the cerebrum defined in the
    config file. You can create your own runner for a complex task like this
     example. 

Notice that there is a simpler runner example called [path_verifier](environment/runners/path_verifier.py) which is used to
  stream camera feeds while moving the robotic arm along all poses for
   calibration using two treads. 
   
It shows another point of creating your runner: by doing so, you can decouple
 the setting for actual equipment from the setting for actual tasks which
  makes your config more clear and therefore easier to manage.


## Built With

I borrowed code from many repos:

* [VedaSeg](https://github.com/Media-Smart/vedaseg)
* [mmdetection](https://github.com/open-mmlab/mmdetection)
* [urx](https://github.com/SintefManufacturing/python-urx) 
* [Pose-Estimation-for-Sensor-Calibration](https://github.com/eayvali/Pose-Estimation-for-Sensor-Calibration)

## Contributing

You are welcome to enlarge the list of supported devices or useful cortex
 functional or even new cortex and more during your project.
 
## Authors

* **Tianhe Wang** - *Initial work* - [Wsnhg](https://github.com/DarthThomas)

## License

TBD

## Acknowledgments

* Hat tip to anyone whose code was used
* Guidance and advice from Hongxiang Cai ([@hxcai](http://github.com/hxcai)), 
Yichao Xiong ([@mileistone](https://github.com/mileistone))
* Helpful discussion with Tianze Rong & 
Shuhan Zhang([@Kuro96](https://github.com/Kuro96))
* Other constructive code reviewers from [Media-Smart-AILab](https://github.com/Media-Smart-AILab)
