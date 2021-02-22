import logging

import numpy as np

from .registry import POSTERIOR_PARIETAL_CORTEX

EPS = 0.00001
FLOAT_EPS_4 = np.finfo(float).eps * 4.0


@POSTERIOR_PARIETAL_CORTEX.register_module
class AxyBSolver:
    """
    Solves for  X and Y in AX=YB from a set of (A,B) paired measurements. (
    Ai,Bi) are absolute pose measurements with known correspondence
        A: (4x4xn)
        X: (4x4) - unknown
        Y: (4x4) - unknown
        B: (4x4xn)
    n: number of measurements
    """
    logger = logging.getLogger(__name__)

    def __init__(self, eps=EPS, float_eps=FLOAT_EPS_4, self_check=True):
        self.eps = eps  # would be used for later-on methods
        self.float_eps = float_eps  # would be used for later-on methods
        self.self_check = self_check
        self.logger.debug(f"Initialized {self.__class__.__name__}:\n"
                          f" {self.__repr__()}")

    def __repr__(self):
        msg = (f"A AX=YB solver with:\n    eps: {self.eps}, "
               f"float_eps: {self.float_eps}, self_check: {self.self_check}\n")
        return msg

    def by_kronecker_product(self, mat_a, mat_b):
        """
        Args:
            mat_a: A: (4x4xn)
            mat_b: B: (4x4xn)

        Returns:
            x_est: estimated X
            y_est: estimated Y
            y_est_check: Y from A@x_est=YB
            error_stats: error metric between y_est and y_est_check
        """
        n = mat_a.shape[2]
        T = np.zeros([9, 9])
        x_est = np.eye(4)
        y_est = np.eye(4)

        # Permutate A and B to get gross motions
        idx = np.random.permutation(n)
        mat_a = mat_a[:, :, idx]
        mat_b = mat_b[:, :, idx]

        for i in range(n - 1):
            rot_a = mat_a[0:3, 0:3, i]
            rot_b = mat_b[0:3, 0:3, i]
            T += np.kron(rot_b, rot_a)

        U, S, Vt = np.linalg.svd(T)
        xp = Vt.T[:, 0]
        yp = U[:, 0]
        # F: fortran/matlab reshape order
        X = np.reshape(xp, (3, 3), order="F")
        Xn = (np.sign(np.linalg.det(X)) / np.abs(np.linalg.det(X)) ** (
                1 / 3)) * X
        # re-orthogonalize to guarantee that they are indeed rotations.
        U_n, S_n, Vt_n = np.linalg.svd(Xn)
        X = np.matmul(U_n, Vt_n)

        Y = np.reshape(yp, (3, 3),
                       order="F")  # F: fortran/matlab reshape order
        Yn = (np.sign(np.linalg.det(Y)) / np.abs(np.linalg.det(Y)) ** (
                1 / 3)) * Y
        U_yn, S_yn, Vt_yn = np.linalg.svd(Yn)
        Y = np.matmul(U_yn, Vt_yn)

        a_est = np.zeros([3 * n, 6])
        b_est = np.zeros([3 * n, 1])
        for i in range(n - 1):
            a_est[3 * i:3 * i + 3, :] = np.concatenate((-mat_a[0:3, 0:3, i],
                                                        np.eye(3)), axis=1)
            b_est[3 * i:3 * i + 3, :] = np.transpose(mat_a[0:3, 3, i] -
                                                     np.matmul(
                                                         np.kron(
                                                             mat_b[0:3, 3, i].T,
                                                             np.eye(3)),
                                                         np.reshape(
                                                             Y, (9, 1),
                                                             order="F")).T)

        t_est_np = np.linalg.lstsq(a_est, b_est, rcond=None)
        if t_est_np[2] < a_est.shape[1]:  # a_est.shape[1]=6
            print('Rank deficient')
        t_est = t_est_np[0]
        x_est[0:3, 0:3] = X
        x_est[0:3, 3] = t_est[0:3].T
        y_est[0:3, 0:3] = Y
        y_est[0:3, 3] = t_est[3:6].T
        if not self.self_check:
            return x_est, y_est
        # verify Y_est using rigid_registration
        y_est_check, error_stats = self.rigid_registration(mat_a, x_est, mat_b)
        return x_est, y_est, y_est_check, error_stats

    @staticmethod
    def rigid_registration(mat_a, mat_x, mat_b):
        """
        solves for Y in YB=AX with known A, B, X
        Args:
            mat_a: A: (4x4xn)
            mat_b: B: (4x4xn)
            mat_x: X: (4x4)

        Returns:

            Y_est: Y in YB=AX with known A, B, X
            error_stats: error metric (mean, std)
        """
        n = mat_a.shape[2]
        AX = np.zeros(mat_a.shape)
        AXp = np.zeros(mat_a.shape)
        Bp = np.zeros(mat_b.shape)
        Y_est = np.eye(4)

        error_stats = np.zeros((2, 1))

        for i in range(n):
            AX[:, :, i] = np.matmul(mat_a[:, :, i], mat_x)

        # Centroid of transformations t and that
        t = 1 / n * np.sum(AX[0:3, 3, :], 1)
        that = 1 / n * np.sum(mat_b[0:3, 3, :], 1)
        AXp[0:3, 3, :] = AX[0:3, 3, :] - np.tile(t[:, np.newaxis], (1, n))
        Bp[0:3, 3, :] = mat_b[0:3, 3, :] - np.tile(that[:, np.newaxis], (1, n))

        [i, j, k] = AX.shape  # 4x4xn
        # Convert AX and B to 2D arrays
        AXp_2D = AXp.reshape((i, j * k))  # now it is 4x(4xn)
        Bp_2D = Bp.reshape((i, j * k))  # 4x(4xn)
        # calculates the best rotation
        U, S, Vt = np.linalg.svd(np.matmul(Bp_2D[0:3, :], AXp_2D[0:3, :].T))
        rot_eat = np.matmul(Vt.T, U.T)
        # special reflection case
        if np.linalg.det(rot_eat) < 0:
            print('Warning: Y_est returned a reflection')
            rot_eat = np.matmul(Vt.T, np.matmul(np.diag([1, 1, -1]), U.T))
        # Calculates the best transformation
        t_est = t - np.dot(rot_eat, that)
        Y_est[0:3, 0:3] = rot_eat
        Y_est[0:3, 3] = t_est
        # Calculate registration error
        pYB = (np.matmul(rot_eat, mat_b[0:3, 3, :]) +
               np.tile(t_est[:, np.newaxis], (1, n)))  # 3xn
        pAX = AX[0:3, 3, :]

        reg_error = np.linalg.norm(pAX - pYB, axis=0)  # 1xn
        error_stats[0] = np.mean(reg_error)
        error_stats[1] = np.std(reg_error)
        return Y_est, error_stats
