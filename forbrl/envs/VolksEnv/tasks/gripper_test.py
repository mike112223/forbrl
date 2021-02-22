import argparse
import os
import sys
import time

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(cur_path, '../../volksenv'))

from environment.assembler import assemble


def parse_args():
    parser = argparse.ArgumentParser(description='CalHCam test')

    parser.add_argument('--config', help="config file for offset test",
                        default='configs/gripper_test.py')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    env = assemble(args.config)
    # env.gripper_1.close()
    # time.sleep(1)
    env.gripper_1.open()
    time.sleep(1)


if __name__ == '__main__':
    main()
