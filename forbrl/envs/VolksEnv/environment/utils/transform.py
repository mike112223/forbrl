
import math

import numpy as np


def img_pixel_to_cam_coor(pix_x, pix_y, cam_pts_z, camera_intrinsics):

    pix = np.concatenate((pix_x[:, :, np.newaxis], pix_y[:, :, np.newaxis]), axis=2)

    f = np.array([camera_intrinsics[0][2], camera_intrinsics[1][2]])
    u = np.array([camera_intrinsics[0][0], camera_intrinsics[1][1]])
    z = np.tile(cam_pts_z[:, :, np.newaxis], 2)

    cam_pts = (pix - f) * (z / u)

    cam_pts = np.concatenate([cam_pts, cam_pts_z[:, :, np.newaxis]], axis=2)

    return cam_pts


def cam_coor_to_robot_coor(surface_pts, cam_pose):
    surface_pts = np.transpose(
        np.dot(cam_pose[0:3, 0:3], np.transpose(surface_pts)) +
        np.tile(cam_pose[0:3, 3:], (1, surface_pts.shape[0])))
    return surface_pts


def get_point_heightmap(point, point_depth,
                        cam_intrinsics, cam_pose,
                        workspace_limits, heightmap_resolution):

    cam_pts_x, cam_pts_y = img_pixel_to_cam_coor(point[1], point[0], point_depth, cam_intrinsics)
    cam_pts = np.array([cam_pts_x, cam_pts_y, point_depth]).reshape(1, -1)
    surface_pts = cam_coor_to_robot_coor(cam_pts, cam_pose)

    # Create orthographic top-down-view RGB-D heightmaps
    heightmap_pix_x = int((surface_pts[:, 0] - workspace_limits[0][0]) / heightmap_resolution)
    heightmap_pix_y = int((surface_pts[:, 1] - workspace_limits[1][0]) / heightmap_resolution)
    heightmap_point = (heightmap_pix_y, heightmap_pix_x)

    depth_heightmap = surface_pts[:, 2]
    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom

    return heightmap_point, depth_heightmap


def get_pointcloud(color_img, depth_img, camera_intrinsics):

    # Get depth image size
    im_h, im_w = depth_img.shape
    cam_pts_z = depth_img.copy()

    # Project depth into 3D point cloud in camera coordinates
    pix_x, pix_y = np.meshgrid(
        np.linspace(0, im_w - 1, im_w),
        np.linspace(0, im_h - 1, im_h)
    )

    cam_pts = img_pixel_to_cam_coor(pix_x, pix_y, cam_pts_z, camera_intrinsics)

    # Reshape image into colors for 3D point cloud
    cam_pts = cam_pts.reshape(-1, 3)
    rgb_pts = color_img.reshape(-1, 3)

    return cam_pts, rgb_pts


def get_heightmap(color_img, depth_img,
                  cam_intrinsics, cam_pose,
                  workspace_limits, heightmap_resolution):

    # Compute heightmap size
    heightmap_size = np.round((
        (workspace_limits[1][1] - workspace_limits[1][0]) / heightmap_resolution,
        (workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_resolution)
    ).astype(int)

    # Get 3D point cloud from RGB-D images
    surface_pts, color_pts = get_pointcloud(color_img, depth_img, cam_intrinsics)

    # Transform 3D point cloud from camera coordinates to robot coordinates
    surface_pts = cam_coor_to_robot_coor(surface_pts, cam_pose)

    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind = np.logical_and(
        np.logical_and(
            np.logical_and(
                np.logical_and(surface_pts[:, 0] >= workspace_limits[0][0],
                               surface_pts[:, 0] < workspace_limits[0][1]),
                surface_pts[:, 1] >= workspace_limits[1][0]),
            surface_pts[:, 1] < workspace_limits[1][1]),
        surface_pts[:, 2] < workspace_limits[2][1])
    surface_pts = surface_pts[heightmap_valid_ind]
    color_pts = color_pts[heightmap_valid_ind]

    # Create orthographic top-down-view RGB-D heightmaps
    color_heightmap = np.zeros(list(heightmap_size) + [3], dtype=np.uint8)
    depth_heightmap = np.zeros(heightmap_size)

    heightmap_pix = np.floor((surface_pts[:, (0, 1)] -
                              workspace_limits[(0, 1), 0]) /
                             heightmap_resolution).astype(int)
    color_heightmap[heightmap_pix[:, 1], heightmap_pix[:, 0]] = color_pts
    depth_heightmap[heightmap_pix[:, 1], heightmap_pix[:, 0]] = surface_pts[:, 2]
    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap == -z_bottom] = np.nan

    return color_heightmap, depth_heightmap


# Get rotation matrix from euler angles
def euler2rotm(theta):
    r_x = np.array([
        [1, 0, 0],
        [0, math.cos(theta[0]), -math.sin(theta[0])],
        [0, math.sin(theta[0]), math.cos(theta[0])]
    ])
    r_y = np.array([[
        math.cos(theta[1]), 0, math.sin(theta[1])],
        [0, 1, 0],
        [-math.sin(theta[1]), 0, math.cos(theta[1])]
    ])
    r_z = np.array([
        [math.cos(theta[2]), -math.sin(theta[2]), 0],
        [math.sin(theta[2]), math.cos(theta[2]), 0],
        [0, 0, 1]
    ])
    r = np.dot(r_z, np.dot(r_y, r_x))
    return r


# Checks if a matrix is a valid rotation matrix.
def isRotm(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
def rotm2euler(R):

    assert(isRotm(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def angle2rotm(angle, axis, point=None):
    # Copyright (c) 2006-2018, Christoph Gohlke

    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis = axis / np.linalg.norm(axis)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(axis, axis) * (1.0 - cosa)
    axis *= sina
    R += np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0]
    ])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:

        # Rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def rotm2angle(R):
    # From: euclideanspace.com

    epsilon = 0.01 # Margin to allow for rounding errors
    epsilon2 = 0.1 # Margin to distinguish between 0 and 180 degrees

    assert(isRotm(R))

    if (
        (abs(R[0][1] - R[1][0]) < epsilon) and
        (abs(R[0][2] - R[2][0]) < epsilon) and
        (abs(R[1][2] - R[2][1]) < epsilon)
    ):
        # Singularity found
        # First check for identity matrix which must have +1 for all terms in leading diagonaland zero in other terms
        if (
            (abs(R[0][1] + R[1][0]) < epsilon2) and
            (abs(R[0][2] + R[2][0]) < epsilon2) and
            (abs(R[1][2] + R[2][1]) < epsilon2) and
            (abs(R[0][0] + R[1][1] + R[2][2] - 3) < epsilon2)
        ):
            # this singularity is identity matrix so angle = 0
            return [0, 1, 0, 0] # zero angle, arbitrary axis

        # Otherwise this singularity is angle = 180
        angle = np.pi
        xx = (R[0][0] + 1) / 2
        yy = (R[1][1] + 1) / 2
        zz = (R[2][2] + 1) / 2
        xy = (R[0][1] + R[1][0]) / 4
        xz = (R[0][2] + R[2][0]) / 4
        yz = (R[1][2] + R[2][1]) / 4
        if ((xx > yy) and (xx > zz)): # R[0][0] is the largest diagonal term
            if (xx < epsilon):
                x = 0
                y = 0.7071
                z = 0.7071
            else:
                x = np.sqrt(xx)
                y = xy / x
                z = xz / x
        elif (yy > zz): # R[1][1] is the largest diagonal term
            if (yy < epsilon):
                x = 0.7071
                y = 0
                z = 0.7071
            else:
                y = np.sqrt(yy)
                x = xy / y
                z = yz / y
        else: # R[2][2] is the largest diagonal term so base result on this
            if (zz < epsilon):
                x = 0.7071
                y = 0.7071
                z = 0
            else:
                z = np.sqrt(zz)
                x = xz / z
                y = yz / z
        return [angle, x, y, z] # Return 180 deg rotation

    # As we have reached here there are no singularities so we can handle normally
    s = np.sqrt(
        (R[2][1] - R[1][2]) * (R[2][1] - R[1][2]) + (R[0][2] - R[2][0]) *
        (R[0][2] - R[2][0]) + (R[1][0] - R[0][1]) * (R[1][0] - R[0][1])
    ) # used to normalise
    if (abs(s) < 0.001):
        s = 1

    # Prevent divide by zero, should not happen if matrix is orthogonal and should be
    # Caught by singularity test above, but I've left it in just in case
    angle = np.arccos(( R[0][0] + R[1][1] + R[2][2] - 1) / 2)
    x = (R[2][1] - R[1][2]) / s
    y = (R[0][2] - R[2][0]) / s
    z = (R[1][0] - R[0][1]) / s
    return [angle, x, y, z]


# from csys to tool coor system, intrinsic rotation XYZ
def euler2rotm_ixyz(theta):
    r_x = np.array([
        [1, 0, 0],
        [0, math.cos(theta[0]), -math.sin(theta[0])],
        [0, math.sin(theta[0]), math.cos(theta[0])]
    ])
    r_y = np.array([
        [math.cos(theta[1]), 0, math.sin(theta[1])],
        [0, 1, 0],
        [-math.sin(theta[1]), 0, math.cos(theta[1])]
    ])
    r_z = np.array([
        [math.cos(theta[2]), -math.sin(theta[2]), 0],
        [math.sin(theta[2]), math.cos(theta[2]), 0],
        [0, 0, 1]
    ])
    r = np.dot(r_x, np.dot(r_y, r_z))
    return r


def rotm2euler_ixyz(R):

    assert(isRotm(R))

    sy = math.sqrt(R[1, 2]**2 + R[2, 2]**2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(-R[1, 2], R[2, 2])
        y = math.atan2(R[0, 2], sy)
        z = math.atan2(-R[0, 1], R[0, 0])

    else:
        x = math.atan2(R[2, 1], R[1, 1])
        y = math.atan2(R[0, 2], sy)
        z = 0

    return np.array([x, y, z])
