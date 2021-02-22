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
    parser.add_argument('--no_show', help="disable image showing",
                        action='store_true')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    env = assemble(args.config)
    cam = env.camera1
    cam.start()
    calhcam = env.cal_h_cam1

    axis = np.float32([[0, 0, 0],
                       [0.285, 0, 0],
                       [0.285, 0.41, 0],
                       [0, 0.41, 0]]).reshape(-1, 3)

    try:
        while True:
            color_img, depth_img = cam.capture()
            cal2cam = calhcam(color_img)
            if cal2cam is not None:
                cam2cal = np.linalg.inv(cal2cam)
                cal_o = np.array([[0, 0, 0, 1]]).T
                cam_o = cam2cal @ cal_o
                print(f"Current distance: "
                      f"{np.linalg.norm(cam_o.squeeze()[:3]) * 100:.2f} cm "
                      f"with height {cam_o.squeeze()[2] * 100:.2f} cm")
                calhcam.project_polygon(color_img, axis)
                calhcam.project_corners(color_img)
            if not args.no_show:
                cv2.namedWindow("Board placement", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("Board placement", color_img)

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        cam.stop()


if __name__ == '__main__':
    main()
