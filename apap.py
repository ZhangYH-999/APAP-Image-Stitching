from constant import *

import numpy as np
import cv2


class APAP:
    def __init__(self, canvas_shape, image1, image2, sigma=sigma, gamma=gamma):
        self.sigma = sigma
        self.gamma = gamma
        self.canvas_h = canvas_shape[0]
        self.canvas_w = canvas_shape[1]
        self.image_1 = image1
        self.image_2 = image2
        self.image1_shape = self.image_1.shape
        self.image2_shape = self.image_2.shape

    def construct_A(self, con_pts1, con_pts2):
        """
        :param con_pts1:   pts shape:(n, 3)    eg: (x, y, 1)
        :param con_pts2:
        :return:
        """
        num_points = con_pts1.shape[0]
        A = np.zeros((num_points*2, 9), dtype=np.float)
        for k in range(num_points):
            p1 = con_pts1[k]
            p2 = con_pts2[k]
            A[2*k, 0:3] = p1*p2[2]
            A[2*k, 6:9] = -p1*p2[0]

            A[2*k+1, 3:6] = p1*p2[2]
            A[2*k+1, 6:9] = -p1*p2[1]
        return A

    def normalise_2d(self, points):
        """
        :param points: shape (n, 2)
        :return: T, normal points (n, 2)
        """
        center = np.mean(points, axis=0)     # (c_x, c_y)
        square = np.square(points - center)  # (x - c_x, y - c_y)
        sum = np.sum(square, axis=1)         # x^2 + y^2             shape:(num, 1)
        mean = np.mean(np.sqrt(sum))         # sqrt(x^2 + y^2)
        scale = np.sqrt(2) / mean
        T = np.array([[scale, 0, -scale * center[0]],
                      [0, scale, -scale * center[1]],
                      [0, 0, 1]], dtype=np.float)

        origin_point = np.copy(points)
        padding = np.ones(points.shape[0])
        origin_point = np.column_stack((origin_point, padding))  # shape:(num, 3)

        normal_points = T.dot(origin_point.T)   # shape:(3, 3) dot (3, num) = (3, num)
        normal_points = normal_points.T[:, :2]  # shape:(num, 2)
        return T, normal_points

    def get_conditioner(self, points):
        """
        :param points: shape (n, 2)
        :return:
        """
        sample_n, _ = points.shape
        calculate = np.expand_dims(points, 0)
        mean_pts, std_pts = cv2.meanStdDev(calculate)
        mean_x, mean_y = np.squeeze(mean_pts)
        std_pts = np.squeeze(std_pts)
        std_pts = std_pts * std_pts * sample_n / (sample_n - 1)
        std_pts = np.sqrt(std_pts)
        std_x, std_y = std_pts
        std_x = std_x + (std_x == 0)
        std_y = std_y + (std_y == 0)
        norm_x = np.sqrt(2) / std_x
        norm_y = np.sqrt(2) / std_y
        T = np.array([[norm_x, 0, (-norm_x * mean_x)],
                      [0, norm_y, (-norm_y * mean_y)],
                      [0, 0, 1]], dtype=np.float)
        return T

    def condition_2d(self, c,  points):
        """
        :param points: (n, 3) or (n, 2)
        :param c: (3, 3)
        :return: (n, 3)
        """
        points = points.T
        if points.shape[0] is 2:        # (2, n)
            points = np.row_stack((points, np.ones(points.shape[1])))
        # (3, 3) * (3, n) = (3, n)
        return np.dot(c, points).T

    def local_homography(self, src_points, dst_points, mesh_vertices):
        """
        :param src_points:  shape:(n, 2)
        :param dst_points:
        :param mesh_vertices:  shape:(mesh_size, mesh_size, 2)
        :return: local H  shape:(mesh_size * mesh_size, 3, 3)
        """
        num_points = src_points.shape[0]
        mesh_h, mesh_w, _ = mesh_vertices.shape

        # normalise points
        N1, normal_point1 = self.normalise_2d(src_points)   # (n, 2)
        N2, normal_point2 = self.normalise_2d(dst_points)

        # condition points
        C1 = self.get_conditioner(normal_point1)
        C2 = self.get_conditioner(normal_point2)
        condition_point1 = self.condition_2d(C1, normal_point1)     # (n, 3)
        condition_point2 = self.condition_2d(C2, normal_point2)
        # get A
        A = self.construct_A(condition_point1, condition_point2)          # (2*n, 9)

        local_homography = np.zeros((mesh_h, mesh_w, 3, 3))

        inv_sigma_2 = 1./(self.sigma ** 2)

        # Gki = exp(-pdist2(Mv(i,:),Kp)./sigma^2)
        # Wi = max(gamma,Gki)
        for i in range(mesh_h):
            for j in range(mesh_w):
                dist = np.sqrt((mesh_vertices[i, j, 0] - src_points[:, 0])**2 + (mesh_vertices[i, j, 1] - src_points[:, 1])**2)
                Weight = np.exp(-dist * inv_sigma_2)   # (n, 1)
                Weight[Weight < self.gamma] = self.gamma
                # W.*A     (2*n,) .* (2*n, 9)
                W_expand = np.reshape(np.repeat(Weight, 2), (-1, 1))
                W_A = W_expand*A
                w, u, vh = cv2.SVDecomp(W_A)
                local_h = vh[-1, :]
                local_h = local_h.reshape((3, 3))
                # de-condition
                local_h = np.linalg.inv(C2).dot(local_h).dot(C1)
                # de-normalise
                local_h = np.linalg.inv(N2).dot(local_h).dot(N1)
                local_h = local_h/local_h[2, 2]
                local_homography[i, j] = local_h
        return local_homography

    def local_warp(self, local_homography, X, Y, offset):
        h2, w2, _ = self.image2_shape
        warp_image = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        for i in range(self.canvas_w):
            idx_x = np.where(i < X)[0][0] - 1
            for j in range(self.canvas_h):
                idx_y = np.where(j < Y)[0][0] - 1
                h = local_homography[idx_y, idx_x, :]
                src_p = np.array([i-offset[0], j-offset[1], 1])
                target_p = np.dot(h, src_p)  # (3, 3) (3, 1)
                target_p /= target_p[2]
                if (0 < target_p[0] < w2) and (0 < target_p[1] < h2):
                    warp_image[j, i, :] = self.image_2[int(target_p[1]), int(target_p[0]), :]
        return warp_image

    def local_warp_beta(self, local_homography, X, Y, offset, gh1, gh2, mask1, mask2):
        h2, w2, _ = self.image2_shape
        warp_image = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        for i in range(self.canvas_w):
            idx_x = np.where(i < X)[0][0] - 1
            for j in range(self.canvas_h):
                idx_y = np.where(j < Y)[0][0] - 1
                h = local_homography[idx_y, idx_x, :]
                weight_2 = mask2[j, i]
                weight_1 = mask1[j, i]
                h = weight_1 * (weight_2 * gh1 + (1 - weight_2) * gh2) + (1 - weight_1) * h

                src_p = np.array([i-offset[0], j-offset[1], 1])
                target_p = np.dot(h, src_p)  # (3, 3) (3, 1)
                target_p /= target_p[2]
                if (0 < target_p[0] < w2) and (0 < target_p[1] < h2):
                    warp_image[j, i, :] = self.image_2[int(target_p[1]), int(target_p[0]), :]
        return warp_image



