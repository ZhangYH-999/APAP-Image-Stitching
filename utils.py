import numpy as np
import cv2


# def calculate_RMSE_beta(offset, feature_point, warped_image_1, warped_image_2):
#     feature_point += offset
#     N = len(feature_point)
#     dist = 0
#     for i in range(N):
#         dist += np.sqrt(np.sum((warped_image_1[int(feature_point[i, 0]), int(feature_point[i, 1]), :] - warped_image_2[int(feature_point[i, 0]), int(feature_point[i, 1]), :])**2, axis=2))
#     emse = np.sqrt(dist / N)
#     return emse


def calculate_RMSE(warped_image_1, warped_image_2):
    mask1 = np.array(np.sum(warped_image_1, axis=-1) > 0, dtype=np.uint8)
    mask2 = np.array(np.sum(warped_image_2, axis=-1) > 0, dtype=np.uint8)
    mask1[mask1 > 0] = 255
    mask2[mask2 > 0] = 255
    ret, mask1 = cv2.threshold(mask1, 200, 255, cv2.THRESH_BINARY)
    ret, mask2 = cv2.threshold(mask2, 200, 255, cv2.THRESH_BINARY)

    overlap_mask = np.array(mask1 & mask2, dtype=np.uint8)
    ret, overlap_mask = cv2.threshold(overlap_mask, 200, 255, cv2.THRESH_BINARY)

    # x, y, w, h = cv2.boundingRect(overlap_mask)
    # res = cv2.rectangle(overlap_mask, (x, y), (x + w, y + h), (255, 255, 255), 1)
    # cv2.imshow("ROI", res)
    # cv2.waitKey()

    count = np.count_nonzero(overlap_mask)
    mask1 = np.expand_dims(mask1, axis=-1)
    mask2 = np.expand_dims(mask2, axis=-1)

    overlap_image_1 = mask1 & mask2 & warped_image_1
    overlap_image_2 = mask1 & mask2 & warped_image_2

    distance = np.sqrt(np.sum((overlap_image_1 - overlap_image_2)**2, axis=2))
    emse = np.sqrt(np.sum(distance)/count)
    return emse


if __name__ == '__main__':
    warped_image_1 = cv2.imread('./TestImage/school/w1.jpg')
    warped_image_2 = cv2.imread('./TestImage/school/w2.jpg')

    print(calculate_RMSE(warped_image_1, warped_image_2))


def global_warp(gh, canvas_shape, image_1, image_2, offset):
    canvas_h, canvas_w = canvas_shape
    h1, w1, _ = image_1.shape
    h2, w2, _ = image_2.shape
    warped_image_2 = np.zeros((canvas_h, canvas_w, 3), dtype="uint8")
    warped_image_2[offset[1]:h2 + offset[1], offset[0]:w2 + offset[0]] = image_2
    warped_image_2 = cv2.warpPerspective(warped_image_2, gh, (warped_image_2.shape[1], warped_image_2.shape[0]))
    warped_image_1 = np.zeros_like(warped_image_2)
    warped_image_1[offset[1]:h1 + offset[1], offset[0]:w1 + offset[0]] = image_1
    global_res = uniform_blend(warped_image_2, warped_image_1)
    return global_res,warped_image_1, warped_image_2


def normalise2dPts(points):
    """
    :param points: surf match points  shape: (n, 2)
    :return:
    """
    c = np.mean(points, axis=0)     # (mean_x, mean_y)
    square = np.square(points - c)  # (x, y) = (x - c_x, y - c_y)
    sum = np.sum(square, axis=1)    # x^2 + y^2             shape:(num, 1)
    mean = np.mean(np.sqrt(sum))    # sqrt(x^2 + y^2)
    scale = np.sqrt(2)/mean
    t = np.array([[scale, 0, -scale * c[0]],
                  [0, scale, -scale * c[1]],
                  [0, 0, 1]], dtype=np.float)
    origin_point = np.copy(points)
    padding = np.ones(points.shape[0])
    origin_point = np.column_stack((origin_point, padding))     # shape:(num, 3)
    new_point = t.dot(origin_point.T)       # shape:(3, 3) dot (3, num) = (3, num)
    new_point = new_point.T[:, :2]          # shape:(num, 2)
    return t, new_point


def get_mesh(size, mesh_size, start=0):
    """
    :param size: final size [width, height]
    :param mesh_size: # of mesh
    :param start: default 0
    :return:
    """
    w, h = size
    x = np.linspace(start, w, mesh_size)
    y = np.linspace(start, h, mesh_size)

    return np.stack([x, y], axis=0)


def get_vertices(size, mesh_size, offsets):
    """
    :param size: final size [width, height]
    :param mesh_size: # of mesh
    :param offsets: [offset_x, offset_y]
    :return:
    """
    w, h = size
    x = np.linspace(0, w, mesh_size)
    y = np.linspace(0, h, mesh_size)
    next_x = x + w / (mesh_size * 2)
    next_y = y + h / (mesh_size * 2)
    next_x, next_y = np.meshgrid(next_x, next_y)
    vertices = np.stack([next_x, next_y], axis=-1)
    vertices -= np.array(offsets)

    return vertices


def uniform_blend(img1, img2):
    mask_1 = np.mean(img1, axis=-1) > 0
    mask_2 = np.mean(img2, axis=-1) > 0

    center = mask_1 & mask_2
    mask = np.expand_dims(center * 0.5, axis=-1)
    mask = np.tile(mask, [1, 1, 3])
    mask[mask == 0] = 1

    result = (img1.astype(np.float64) + img2.astype(np.float64))
    result *= mask
    result = result.astype(np.uint8)

    return result
