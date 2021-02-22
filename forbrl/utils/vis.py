
import os

import cv2
import numpy as np
from scipy import ndimage


def get_pred_vis(preds, color_heightmap, best_idx):

    canvas = None
    num_rotations = preds.shape[0]
    for canvas_row in range(int(num_rotations / 4)):
        tmp_row_canvas = None
        for canvas_col in range(4):
            rotate_idx = canvas_row * 4 + canvas_col
            pred_vis = preds[rotate_idx, :, :].copy()

            pred_vis = np.clip(pred_vis, 0, 1)
            pred_vis.shape = (preds.shape[1], preds.shape[2])
            pred_vis = cv2.applyColorMap(
                (pred_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)

            if rotate_idx == best_idx[0]:
                pred_vis = cv2.circle(
                    pred_vis, (int(best_idx[2]), int(best_idx[1])),
                    7, (0, 0, 255), 2)

            pred_vis = ndimage.rotate(
                pred_vis, rotate_idx * (360.0 / num_rotations),
                reshape=False, order=0)
            bg_img = ndimage.rotate(
                color_heightmap, rotate_idx * (360.0 / num_rotations),
                reshape=False, order=0)
            pred_vis = (0.5 * cv2.cvtColor(bg_img, cv2.COLOR_RGB2BGR) +
                        0.5 * pred_vis).astype(np.uint8)

            if tmp_row_canvas is None:
                tmp_row_canvas = pred_vis
            else:
                tmp_row_canvas = np.concatenate(
                    (tmp_row_canvas, pred_vis), axis=1)

        if canvas is None:
            canvas = tmp_row_canvas
        else:
            canvas = np.concatenate((canvas, tmp_row_canvas), axis=0)

    return canvas

def save_vis(dir, iter, vis, name):
    cv2.imwrite(
        os.path.join(dir, '%06d.%s.png' % (iter, name)), vis)
