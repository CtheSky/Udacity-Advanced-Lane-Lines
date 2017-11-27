def calc_calibration_params():
    """Calculate and return camera matrix and distortion coefficients."""
    import cv2
    import glob
    import numpy as np

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d points in real world space
    img_points = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, filename in enumerate(images):
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret:
            obj_points.append(objp)
            img_points.append(corners)

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image.shape[1::-1], None, None)
    return mtx, dist


def get_calibration_params():
    """Get the calibration params from pickled file.
       If file not exists, compute it and cache it by pickle."""
    import os
    import pickle

    filename = 'calibration_params.p'
    if os.path.exists(filename):
        return pickle.load(open(filename, 'rb'))
    else:
        mtx, dist = calc_calibration_params()
        with open(filename, 'wb') as f:
            pickle.dump((mtx, dist), f, pickle.HIGHEST_PROTOCOL)
        return mtx, dist


def undistort_image(image):
    """Undistort the image using calculated calibration params"""
    import cv2

    mtx, dist = get_calibration_params()
    return cv2.undistort(image, mtx, dist, None, mtx)


if __name__ == '__main__':
    import argparse
    import cv2
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser('Display undistorted calibration image.')
    parser.add_argument('-f', default='camera_cal/calibration1.jpg', help='path to test image')
    args = parser.parse_args()

    img = cv2.imread(args.f)
    dst = undistort_image(img)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.show()
