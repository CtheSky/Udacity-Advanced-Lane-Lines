import numpy as np

from camera_undistort import undistort_image
from threshold import combined_threshold
from perspective_transform import original2bird_eye, invert_matrix
from find_lane_line import get_curvature, find_lane_lines


def process_image(image, debug=False):
    undistorted = undistort_image(image)
    threshold = combined_threshold(undistorted)
    bird_eye = original2bird_eye(threshold)
    lane_line_params = find_lane_lines(bird_eye)
    result = draw_lane_lines(undistorted, lane_line_params)

    if debug:
        f, axarr = plt.subplots(3, 2, figsize=(15, 15))
        axarr[0, 0].imshow(undistorted)
        axarr[0, 0].set_title('Undistorted Image')

        axarr[0, 1].imshow(threshold, cmap='gray')
        axarr[0, 1].set_title('Threshold Image')

        axarr[1, 0].imshow(original2bird_eye(undistorted))
        axarr[1, 0].set_title('Bird Eye Image')

        axarr[1, 1].imshow(bird_eye, cmap='gray')
        axarr[1, 1].set_title('Bird Eye threshold Image')

        axarr[2, 0].imshow(lane_line_params['out_img'])
        axarr[2, 0].plot(lane_line_params['left_fit_x'], lane_line_params['plot_y'], color='yellow')
        axarr[2, 0].plot(lane_line_params['right_fit_x'], lane_line_params['plot_y'], color='yellow')
        axarr[2, 0].set_title('Curvature Image')

        axarr[2, 1].imshow(result)
        axarr[2, 1].set_title('Result Image')
        plt.show()

        print(lane_line_params['curvature'])

    return result


def draw_lane_lines(undistorted, params):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(params['binary_warped']).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([params['left_fit_x'], params['plot_y']]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([params['right_fit_x'], params['plot_y']])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Inverse transform and combine the result with the original image
    new_warp = cv2.warpPerspective(color_warp, invert_matrix, (image.shape[1], image.shape[0]))
    return cv2.addWeighted(undistorted, 1, new_warp, 0.3, 0)


if __name__ == '__main__':
    import argparse
    import cv2
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser('Display processed image.')
    parser.add_argument('-f', default='test6.jpg', help='name of test image')
    args = parser.parse_args()

    image = cv2.imread('test_images/{}'.format(args.f))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = process_image(image, debug=True)
