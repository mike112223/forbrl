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

    parser.add_argument('--config',
                        help="config file for multi components test",
                        default='configs/multi_components_test.py')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    res = assemble(args.config)

    res.camera1.start()
    res.camera2.start()

    try:
        while True:
            for idx, camera in enumerate(['camera1', 'camera2']):
                color_image, depth_image = res[camera].capture()
                grey_color = 153
                depth_image_3d = np.dstack((depth_image,
                                            depth_image,
                                            depth_image))
                bg_removed = np.where(
                    (depth_image_3d > 0.8) | (depth_image_3d <= 0),
                    grey_color, color_image)

                # Show images
                cv2.namedWindow(f"Frame from {camera}", cv2.WINDOW_AUTOSIZE)
                cv2.imshow(f"Frame from {camera}", bg_removed)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        res.camera1.stop()
        res.camera2.stop()


if __name__ == '__main__':
    main()
