mesh_size = 100

image_max_size = 700*700

# RANSAC argument
max_iteration = 500         # the number of iteration of RANSAC
ransac_threshold = 30      # ransac threshold

# APAP argument
gamma = 0.1
sigma = 8.5

# seam
blend_width = 8

# input path
images_path = "./TestImage/square/"
tar_image_path = images_path + "1.jpg"         # image 1
ref_image_path = images_path + "2.jpg"         # image 2

show_process = False

NUM_CLUSTERS = 2
