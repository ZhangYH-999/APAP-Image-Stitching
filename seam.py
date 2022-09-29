import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage import measure, color
import copy
from constant import *
from scipy import ndimage


def find_seam(image_1, image_2):
    mask_1 = np.array(np.sum(image_1, axis=-1) > 0, dtype=np.uint8)
    mask_2 = np.array(np.sum(image_2, axis=-1) > 0, dtype=np.uint8)
    mask_1[mask_1 > 0] = 255
    mask_2[mask_2 > 0] = 255
    ret, mask_1 = cv2.threshold(mask_1, 100, 255, cv2.THRESH_BINARY)
    ret, mask_2 = cv2.threshold(mask_2, 100, 255, cv2.THRESH_BINARY)

    # cv2.imshow("mask_1", mask_1)
    # cv2.waitKey()
    # cv2.imshow("mask_2", mask_2)
    # cv2.waitKey()

    con_mask = mask_1 | mask_2
    overlap_mask = mask_1 & mask_2
    overlap_mask = np.expand_dims(overlap_mask, axis=-1)
    overlap_image_1 = overlap_mask * image_1
    overlap_image_2 = overlap_mask * image_2

    E = calculate_E(overlap_image_1, overlap_image_2)

    # 计算ROI size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    overlap_mask = cv2.dilate(overlap_mask, kernel)  # 膨胀操作
    x, y, w, h = cv2.boundingRect(overlap_mask)
    # print("ROI: ", (y, x), (y + h, x + w))
    # print("h, w: ", h, w)

    # res = cv2.rectangle(overlap_mask, (x, y), (x + w, y + h), (255, 255, 255), 1)
    # cv2.imshow("roi", overlap_mask)
    # cv2.waitKey()

    # 计算接缝的起点与终点
    edge_1 = cv2.Canny(mask_1, 50, 100)
    edge_2 = cv2.Canny(mask_2, 50, 100)
    in_mask = edge_1 & edge_2

    # cv2.imshow("1", edge_1)

    pts = np.argwhere(in_mask > 0)
    km_cluster = KMeans(n_clusters=2)
    cluster_res = km_cluster.fit_predict(pts)

    start_idx = pts[cluster_res == 0]
    start_idx = start_idx[int(len(start_idx)/2)]

    end_idx = pts[cluster_res == 1]
    end_idx = end_idx[int(len(end_idx)/2)]

    reachable = np.zeros((h, w), dtype=np.bool)
    path = np.zeros((h, w))
    cost = np.zeros((h, w))

    # 计算接缝的方向
    if np.abs(start_idx[0] - end_idx[0]) > np.abs(start_idx[1] - end_idx[1]):   # v
        orientation = 1
        if start_idx[0] > end_idx[0]:
            temp = start_idx
            start_idx = end_idx
            end_idx = temp
        # 左上1  上2   右上3
        # print(start_idx, end_idx)
        # print(start_idx[0] - y, start_idx[1] - x)
        # print(end_idx[0] - y, end_idx[1] - x)
        reachable[start_idx[0]-y, start_idx[1]-x] = True
        for i in range(1, h):
            for j in range(w):
                steps = [float('inf'), 0]  #
                # i+y j+x
                # i, j
                if overlap_mask[i+y, j+x] > 0:
                    if j - 1 >= 0 and reachable[i - 1, j - 1] and E[i+y, j+x] + cost[i - 1, j - 1] < steps[0]:  # 左下
                        steps = [E[i+y, j+x] + cost[i - 1, j - 1], 1]
                    if reachable[i - 1, j] and E[i+y, j+x] + cost[i - 1, j] < steps[0]:                         # 下
                        steps = [E[i+y, j+x] + cost[i - 1, j], 2]
                    if j + 1 < w and reachable[i - 1, j + 1] and E[i+y, j+x] + cost[i - 1, j + 1] < steps[0]:   # 右下
                        steps = [E[i+y, j+x] + cost[i - 1, j + 1], 3]
                    if steps[1] != 0:
                        path[i, j] = steps[1]
                        cost[i, j] = steps[0]
                        reachable[i, j] = True
        seam = []
        j = end_idx[1]
        for i in range(end_idx[0], start_idx[0] - 1, -1):
            seam.append((int(i), int(j)))
            direct = path[int(i - y), int(j - x)] - 2
            j += direct
    else:   # h
        orientation = 0
        if start_idx[1] > end_idx[1]:
            temp = start_idx
            start_idx = end_idx
            end_idx = temp
        # 左上1  上2   右上3
        reachable[start_idx[0]-y, start_idx[1]-x] = True
        for j in range(1, w):
            for i in range(h):
                steps = [float('inf'), 0]
                # i+y j+x
                # i, j
                if overlap_mask[i+y, j+x] > 0:
                    if i - 1 >= 0 and reachable[i-1, j-1] and E[i+y, j+x] + cost[i-1, j-1] < steps[0]:
                        steps = [E[i+y, j+x] + cost[i-1, j-1], 1]
                    if reachable[i, j-1] and E[i+y, j+x] + cost[i, j-1] < steps[0]:
                        steps = [E[i+y, j+x] + cost[i, j-1], 2]
                    if i + 1 < h and reachable[i+1, j-1] and E[i+y, j+x] + cost[i+1, j-1] < steps[0]:
                        steps = [E[i+y, j+x] + cost[i+1, j-1], 3]
                    if steps[1] != 0:
                        path[i, j] = steps[1]
                        cost[i, j] = steps[0]
                        reachable[i, j] = True
        seam = []
        i = end_idx[0]
        for j in range(end_idx[1], start_idx[1] - 1, -1):
            seam.append((int(i), int(j)))
            direct = path[int(i - y), int(j - x)] - 2
            i += direct

    seam_mask = copy.deepcopy(con_mask)

    for pt in seam:
        seam_mask[pt[0], pt[1]] = 0

    labels, n = measure.label(seam_mask, connectivity=1, return_num=True)
    # 获取image1的mask
    res_mask_1 = np.zeros_like(labels, dtype=np.float)
    for i in range(n+1):
        res_mask_1[labels == i] = 1
        if np.sum(res_mask_1[(np.array(cv2.dilate(mask_1, kernel)) == 0) * (res_mask_1 > 0)]) == 0:
            break
        res_mask_1[labels == i] = 0
    # 获取image2的mask
    res_mask_2 = np.zeros_like(labels, dtype=np.float)
    for i in range(n + 1):
        res_mask_2[labels == i] = 1
        if np.sum(res_mask_2[(np.array(cv2.dilate(mask_2, kernel)) == 0) * (res_mask_2 > 0)]) == 0:
            break
        res_mask_2[labels == i] = 0

    # # 羽化融合
    test_pt = seam[0]
    if orientation == 1:
        if res_mask_1[test_pt[0], test_pt[1]-1] > 0:
            for pt in seam:
                res_mask_1[pt[0], pt[1] - blend_width:pt[1] + blend_width] = np.linspace(0, 1, blend_width*2)[::-1]
            for pt in seam:
                res_mask_2[pt[0], pt[1] - blend_width:pt[1] + blend_width] = np.linspace(0, 1, blend_width*2)
        else:
            for pt in seam:
                res_mask_2[pt[0], pt[1] - blend_width:pt[1] + blend_width] = np.linspace(0, 1, blend_width*2)[::-1]
            for pt in seam:
                res_mask_1[pt[0], pt[1] - blend_width:pt[1] + blend_width] = np.linspace(0, 1, blend_width*2)
    else:
        if res_mask_1[test_pt[0]-1, test_pt[1]] > 0:
            for pt in seam:
                res_mask_1[pt[0]-blend_width:pt[0]+blend_width, pt[1]] = np.linspace(0, 1, blend_width*2)[::-1]
            for pt in seam:
                res_mask_2[pt[0]-blend_width:pt[0]+blend_width, pt[1]] = np.linspace(0, 1, blend_width*2)
        else:
            for pt in seam:
                res_mask_2[pt[0]-blend_width:pt[0]+blend_width, pt[1]] = np.linspace(0, 1, blend_width*2)[::-1]
            for pt in seam:
                res_mask_1[pt[0]-blend_width:pt[0]+blend_width, pt[1]] = np.linspace(0, 1, blend_width*2)

    res_mask_1 = np.expand_dims(res_mask_1, axis=-1)
    res_mask_2 = np.expand_dims(res_mask_2, axis=-1)

    res_image1 = res_mask_1 * image_1
    res_image2 = res_mask_2 * image_2
    res_image = res_image1 + res_image2
    cv2.imshow("res_image", np.array(res_image, dtype=np.uint8))
    cv2.waitKey()
    return res_image


def calculate_E(image_1, image_2):
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    Sx = np.array([[-2, 0, 2], [-1, 0, 1], [-2, 0, 2]])
    Sy = np.array([[-2, -1, 2], [0, 0, 0], [2, 1, 2]])

    image_x_1 = cv2.filter2D(image_1, -1, Sx)
    image_y_1 = cv2.filter2D(image_1, -1, Sy)

    image_x_2 = cv2.filter2D(image_2, -1, Sx)
    image_y_2 = cv2.filter2D(image_2, -1, Sy)

    E_color = (image_1 - image_2)**2
    E_geometry = (image_x_1 - image_x_2) * (image_y_1 - image_y_2)
    E = E_color + E_geometry
    return E.astype(float)


# test
if __name__ == "__main__":
    warped_image_1 = cv2.imread("./warped_image1.jpg")
    warped_image_2 = cv2.imread("./warped_image2.jpg")
    find_seam(warped_image_1, warped_image_2)



