import argparse
import os
import sys

import cv2
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(cur_path, '../../volksenv'))

from environment.assembler import assemble


def parse_args():
    parser = argparse.ArgumentParser(description='CalHCam test')

    parser.add_argument('--config', help="config file for offset test",
                        default='configs/offset_test.py')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    env = assemble(args.config)
    cam = env.camera1
    cam.start()
    print(cam.intrinsics)


if __name__ == '__main__':
    main()
