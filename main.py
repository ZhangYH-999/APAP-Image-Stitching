import time

import matplotlib.pyplot as plt

from matcher import *
from apap import *
from utils import *
from seam import *

if __name__ == "__main__":
    # ----------------
    #  Read images
    # ----------------
    print("> Reading images...")
    start = time.time()
    image_1 = cv2.imread(tar_image_path)
    image_2 = cv2.imread(ref_image_path)
    h1, w1, _ = image_1.shape
    h2, w2, _ = image_2.shape
    if h1*w1 >= image_max_size:
        radio = np.sqrt(image_max_size/(h1*w1))
        h1, w1 = int(radio*h1), int(radio*w1)
        image_1 = cv2.resize(image_1, (w1, h1))
    if h2*w2 >= image_max_size:
        radio = np.sqrt(image_max_size / (h2 * w2))
        h2, w2 = int(radio*h2), int(radio*w2)
        image_2 = cv2.resize(image_2, (w2, h2))

    end = time.time()
    print("--- done( %f s ): image_1: " % (end-start), (h1, w1), " image_2: ", (h2, w2))

    # --------------------------------
    # Keypoint detection and matching
    # --------------------------------
    matcher = FeatureMatcher('sift')
    print("> Detecting and Matching keypoint...")
    start = time.time()
    match_pts1, match_pts2 = matcher.run(image_1, image_2)
    end = time.time()

    # plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    # plt.scatter(match_pts1[:, 0], match_pts1[:, 1], c='red', s=5)
    # plt.axis('off')
    # plt.show()
    #
    # plt.imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
    # plt.scatter(match_pts2[:, 0], match_pts2[:, 1], c='blue', s=5)
    # plt.axis('off')
    # plt.show()

    print("--- done( %f s ): shape of match points: " % (end-start), match_pts1.shape)

    # ---------------------------------------
    # RANSAC and calculate global Homography
    # ---------------------------------------
    print("> RANSAC and calculate global homography...")
    start = time.time()
    gh, status = cv2.findHomography(match_pts2, match_pts1,  cv2.RANSAC, ransac_threshold, maxIters=max_iteration)
    status = np.reshape(np.array(status, dtype=np.bool), (-1,))
    final_match_pts1, final_match_pts2 = match_pts1[status, :], match_pts2[status, :]   # (n, 2)
    end = time.time()
    print("--- done( %f s ): shape of final match points: " % (end-start), final_match_pts1.shape)

    # plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    # plt.scatter(match_pts1[:, 0], match_pts1[:, 1], c='red', s=5)
    # plt.scatter(final_match_pts1[:, 0], final_match_pts1[:, 1], c='yellow', s=5)
    # plt.axis('off')
    # plt.show()
    #
    # plt.imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
    # plt.scatter(match_pts2[:, 0], match_pts2[:, 1], c='red', s=5)
    # plt.scatter(final_match_pts2[:, 0], final_match_pts2[:, 1], c='yellow', s=5)
    # plt.axis('off')
    # plt.show()

    # ----------------------
    # Prepare mesh grid
    # ----------------------
    print("> Preparing mesh...")
    start = time.time()
    origin_corners_2 = [np.array([0, 0, 1], dtype=np.float64),      # TL
                        np.array([w2, 0, 1], dtype=np.float64),     # TR
                        np.array([0, h2, 1], dtype=np.float64),     # BL
                        np.array([w2, h2, 1], dtype=np.float64)]    # BR
    project_corners_2 = []
    for corner in origin_corners_2:
        vec = np.dot(gh, corner)
        x, y = vec[0]/vec[2], vec[1]/vec[2]
        project_corners_2.append([x, y])

    project_corners_2 = np.array(project_corners_2, dtype=np.int)
    canvas_w = max(np.max(project_corners_2[:, 0]), w1) - min(np.min(project_corners_2[:, 0]), 0)
    canvas_h = max(np.max(project_corners_2[:, 1]), h1) - min(np.min(project_corners_2[:, 1]), 0)
    offset = [-min(np.min(project_corners_2[:, 0]), 0), - min(np.min(project_corners_2[:, 1]), 0)]

    mesh = get_mesh((canvas_w, canvas_h), mesh_size + 1)
    mesh_vertices = get_vertices((canvas_w, canvas_h), mesh_size, (offset[0], offset[1]))
    end = time.time()
    print("--- done( %f s ): offset x, y: " % (end - start), offset, "  canvas shape: ", (canvas_h, canvas_w))

    # ----------------------
    # K-Means
    # ----------------------
    # init cluster center
    mean_x = np.mean(final_match_pts1[:, 0], axis=0)
    init_center_p = np.array([[mean_x, 0], [mean_x, h1]])

    km_cluster = KMeans(n_clusters=NUM_CLUSTERS, init=init_center_p, n_init=1)
    cluster_res = km_cluster.fit_predict(final_match_pts1)
    group1_match_pts = final_match_pts1[cluster_res == 0, :] - offset
    group2_match_pts = final_match_pts1[cluster_res == 1, :] - offset
    print("K-means: ", len(group1_match_pts), len(group2_match_pts))

    # calculate gh1, gh2
    gh1, _ = cv2.findHomography(final_match_pts2[cluster_res == 0, :], final_match_pts1[cluster_res == 0, :], cv2.RANSAC, ransac_threshold, maxIters=max_iteration)
    gh2, _ = cv2.findHomography(final_match_pts2[cluster_res == 1, :], final_match_pts1[cluster_res == 1, :], cv2.RANSAC, ransac_threshold, maxIters=max_iteration)
    # global_res, _, _ = global_warp(gh1, (canvas_h, canvas_w), image_1, image_2, offset)
    # cv2.imshow("global H1", global_res)
    #
    # global_res, _, _ = global_warp(gh2, (canvas_h, canvas_w), image_1, image_2, offset)
    # cv2.imshow("global H2", global_res)
    # cv2.waitKey()

    # plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    # plt.scatter(final_match_pts1[cluster_res == 0, 0], final_match_pts1[cluster_res == 0, 1], c='red', s=5)
    # plt.scatter(final_match_pts1[cluster_res == 1, 0], final_match_pts1[cluster_res == 1, 1], c='blue', s=5)
    # plt.axis('off')
    # plt.show()
    #
    # plt.imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
    # plt.scatter(final_match_pts2[cluster_res == 0, 0], final_match_pts2[cluster_res == 0, 1], c='red', s=5)
    # plt.scatter(final_match_pts2[cluster_res == 1, 0], final_match_pts2[cluster_res == 1, 1], c='blue', s=5)
    # plt.axis('off')
    # plt.show()

    # -------------------------------------
    # Show the result of global homography
    # -------------------------------------
    # if show_process:
    global_res, global_1, global_2 = global_warp(gh, (canvas_h, canvas_w), image_1, image_2, offset)
        # cv2.imshow("global H", global_res)
        # cv2.waitKey()

    # ##################################################### #
    global_mask_1 = np.array(np.sum(global_1, axis=-1) > 0, dtype=np.uint8)
    global_mask_2 = np.array(np.sum(global_2, axis=-1) > 0, dtype=np.uint8)

    # cv2.imshow("global_mask_1", global_mask_1)
    # cv2.waitKey()
    mask1 = np.ones((canvas_h, canvas_w))
    mask1[global_mask_1 > 0] = 0
    mask1[global_mask_2 > 0] = 1
    mask1 = ndimage.distance_transform_edt(mask1)
    mask1 = mask1 / np.max(mask1)
    mask1 *= 1.2
    mask1[mask1 > 1] = 1
    # cv2.imshow("mask1", mask1)
    # cv2.waitKey()

    mask2 = np.zeros((canvas_h, canvas_w))
    # for i in range(canvas_h):
    #     for j in range(canvas_w):
    #         x = j - offset[0]
    #         y = i - offset[1]
    #         dist_1 = np.min(np.sqrt((x - group1_match_pts[:, 0])**2 + (y - group1_match_pts[:, 1])**2))
    #         dist_2 = np.min(np.sqrt((x - group2_match_pts[:, 0])**2 + (y - group2_match_pts[:, 1])**2))
    #         mask2[i, j] = dist_1/(dist_1 + dist_2)

    center_horizon = int((np.max(group1_match_pts[:, 1]) + np.min(group2_match_pts[:, 1]))/2 + offset[1]*2 + 0.1*canvas_h)
    mask2[0:center_horizon, :] = 1
    mask2 = ndimage.distance_transform_edt(mask2)
    mask2 = mask2 / np.max(mask2)
    mask2 *= 1.8
    mask2[mask2 > 1] = 1

    # cv2.imshow("mask2", mask2)
    # cv2.waitKey()

    # ##################################################### #

    # --------------------------------
    # APAP: 1. calculate local H
    # --------------------------------
    print("> Calculating local homography...")
    start = time.time()
    apap = APAP((canvas_h, canvas_w), image_1, image_2)

    local_H = apap.local_homography(final_match_pts1, final_match_pts2, mesh_vertices)

    end = time.time()

    print("--- done( %f s ):" % (end - start))

    # --------------------------------
    # APAP: 2. warp image by local homography
    # --------------------------------
    print("> Warping image...")
    start = time.time()
    gh1 = np.linalg.inv(gh1)
    gh1 = gh1 / gh1[2, 2]
    gh2 = np.linalg.inv(gh2)
    gh2 = gh2 / gh2[2, 2]
    # warped_image_2 = apap.local_warp_beta(local_H, mesh[0], mesh[1], offset, gh1, gh2, mask1, mask2)
    warped_image_2_ = apap.local_warp(local_H, mesh[0], mesh[1], offset)

    end = time.time()
    print("--- done( %f s ):" % (end - start))

    # --------------------------------
    # Blending images
    # --------------------------------
    print("Blending image...")
    start = time.time()
    #
    warped_image_1 = np.zeros_like(warped_image_2_, dtype=np.uint8)
    warped_image_1[offset[1]: h1 + offset[1], offset[0]: w1 + offset[0], :] = image_1
    #
    # # cv2.imshow("warped_image_1", warped_image_1)
    # # cv2.imshow("warped_image_2", warped_image_2)
    # # cv2.waitKey()
    #
    # res_image = find_seam(warped_image_1, warped_image_2)
    # res_image = uniform_blend(warped_image_1, warped_image_2)
    apap_res = find_seam(warped_image_1, warped_image_2_)
    # apap_res = uniform_blend(warped_image_1, warped_image_2_)
    end = time.time()
    print("--- done( %f s ):" % (end - start))
    cv2.imwrite(images_path + "/apap_res.jpg", apap_res)
    # cv2.imwrite(images_path + "/our_res.jpg", res_image)

    # cv2.imwrite("./warped_image1.jpg", warped_image_1)
    # cv2.imwrite("./warped_image2.jpg", warped_image_2)

    # cv2.imshow("res_image", res_image)
    # cv2.waitKey()

    print("APAP: ", calculate_RMSE(warped_image_1, warped_image_2_))
    # print("Ours: ", calculate_RMSE(warped_image_1, warped_image_2))