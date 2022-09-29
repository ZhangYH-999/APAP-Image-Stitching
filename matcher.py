import cv2
import numpy as np


class FeatureMatcher:
    def __init__(self, method='surf'):
        self.detector = None
        self.matcher = cv2.BFMatcher()
        self.good_match = []
        self.method = method

        self.kpt1 = None
        self.kpt2 = None
        self.des1 = None
        self.des2 = None
        self.good_ratio = 0.7

    def feature_detect(self, src_image, det_image):

        if str.lower(self.method) is 'surf':
            self.detector = cv2.xfeatures2d.SURF_create(hessianThreshold=200, nOctaves=3, nOctaveLayers=4)
        else:
            self.detector = cv2.xfeatures2d.SIFT_create()

        # RGB to gray
        src_gray_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
        det_gray_image = cv2.cvtColor(det_image, cv2.COLOR_BGR2GRAY)

        self.kpt1, self.des1 = self.detector.detectAndCompute(src_gray_image, None)
        self.kpt2, self.des2 = self.detector.detectAndCompute(det_gray_image, None)

    def match(self, k=2):
        return self.matcher.knnMatch(self.des1, self.des2, k=k)

    def select_good_match(self, raw_match):
        # select good match point
        self.good_match = []
        for a, b in raw_match:
            if a.distance < b.distance * self.good_ratio:
                self.good_match.append([a.trainIdx, a.queryIdx])

        match_kpt1 = np.array([self.kpt1[i].pt for _, i in self.good_match])
        match_kpt2 = np.array([self.kpt2[i].pt for i, _ in self.good_match])
        return match_kpt1, match_kpt2

    def run(self, src_image, det_image):
        # detect feature
        self.feature_detect(src_image, det_image)
        # match
        raw_match = self.match()
        # select good match
        return self.select_good_match(raw_match)


