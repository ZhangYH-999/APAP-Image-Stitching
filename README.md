# APAP-Image-Stitching
the APAP image stitching algorithm in python,

requirement: `OpenCV 3.4.2.16`(important), `numpy` , `scikit`



![image-20220929151230141](README.assets/image-20220929151230141.png)![image-20220929151235342](README.assets/image-20220929151235342.png)

![image-20220929151239532](README.assets/image-20220929151239532.png)

### Image Stitching

#### 1. SIFT

through SIFT algorithm, finding feature points of an image on different scale spaces and matching feature points

<img src="README.assets/wps1.jpg" alt="img" style="zoom: 80%;" /><img src="README.assets/wps2.jpg" alt="img" style="zoom: 80%;" />



#### 2. RANSAC

The RANSAC algorithm correctly classifies "inner" and "outer" points in a set of data containing "outer" points. (red points: outer,   yellow points: inner)

<img src="README.assets/wps3.jpg" alt="img" style="zoom:80%;" /><img src="README.assets/wps4.jpg" alt="img" style="zoom:80%;" />



#### 3. Moving DLT

The general flow of the APAP algorithm is as follows:

1. A global homography matrix is calculated using DLT and SVD to predict the size of the panoramic image.

2. Divide the target image with a fixed grid, calculate the Euclidean distance and weight between each grid centroid and each feature point in the target image, construct the matrix by Moving DTL, and use SVD decomposition to find the local homography matrix of the current grid.

3. Use the local homography  matrix to map the target image into the panoramic canvas to obtain the APAP stitched image.





<img src="README.assets/image-20220929144842818.png" alt="image-20220929144842818" style="zoom:50%;" />



### Image Blending

Seam line fusion algorithm, which calculates the energy value of each pixel in the image using energy function, then finds the seam line with the lowest cumulative energy value using dynamic programming algorithm, finally fuses the two images by feathering the seams.

![image-20220929144944115](README.assets/image-20220929144944115.png)



<img src="README.assets/image-20220929145001505.png" alt="image-20220929145001505" style="zoom:67%;" />

<img src="README.assets/image-20220929145009436.png" alt="image-20220929145009436" style="zoom:67%;" />



### Example:

example 1:

![image-20220929150255166](README.assets/image-20220929150255166.png)![image-20220929150259689](README.assets/image-20220929150259689.png)

<img src="README.assets/image-20220929150305422.png" alt="image-20220929150305422" style="zoom:67%;" />



example 2:

![image-20220929150504348](README.assets/image-20220929150504348.png)![image-20220929150513919](README.assets/image-20220929150513919.png)

![image-20220929150519295](README.assets/image-20220929150519295.png)



### Document Intro

`main.py`: run the program to stitch two images

`constant.py`: define some parameters, like the original images file path

`seam.py`: find the seam line by DP algorithm

`matcher.py`: matcher to detect and match the feature points
