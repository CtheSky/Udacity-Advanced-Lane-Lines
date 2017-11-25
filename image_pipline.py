import numpy as np
import cv2

from camera_undistort import undistort_image
from threshold import combined_threshold
from perspective_transform import original2bird_eye, invert_matrix
from find_lane_line import calc_lane_lines, xm_per_pix


def process_image(image, debug=False):
    undistorted = undistort_image(image)
    threshold = combined_threshold(undistorted)
    bird_eye = original2bird_eye(threshold)
    lane_line_params = calc_lane_lines(bird_eye)
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

    return result


def draw_lane_lines(undistorted, params):
    warp_zero = np.zeros_like(params['binary_warped']).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Fill the part between lane lines with green
    fit_pts_left = np.array([np.transpose(np.vstack([params['left_fit_x'], params['plot_y']]))])
    fit_pts_right = np.array([np.flipud(np.transpose(np.vstack([params['right_fit_x'], params['plot_y']])))])
    fit_pts = np.hstack((fit_pts_left, fit_pts_right))
    cv2.fillPoly(color_warp, np.int_([fit_pts]), (0, 255, 0))

    # Inverse transform and combine the result with the original image
    new_warp = cv2.warpPerspective(color_warp, invert_matrix, undistorted.shape[1::-1])
    image = cv2.addWeighted(undistorted, 1, new_warp, 0.3, 0)

    # Add curvature text to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Left radius of curvature  = {:.2f} m'.format(params['left_curvature']),
                (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'Right radius of curvature = {:.2f} m'.format(params['right_curvature']),
                (50, 80), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Add vehicle position to the image
    left_fit_point = np.float64([[[params['left_fit_x'][-1], params['plot_y'][-1]]]])
    right_fit_point = np.float64([[[params['right_fit_x'][-1], params['plot_y'][-1]]]])
    left_fit_in_original = cv2.perspectiveTransform(left_fit_point, invert_matrix)
    right_fit_in_original = cv2.perspectiveTransform(right_fit_point, invert_matrix)

    lane_mid = .5 * (left_fit_in_original + right_fit_in_original)[0, 0, 0]
    vehicle_mid = image.shape[1] / 2
    dx = (vehicle_mid - lane_mid) * xm_per_pix

    cv2.putText(image, 'Vehicle position : {:.2f} m {} of center'.format(abs(dx), 'left' if dx < 0 else 'right'),
                (50, 110), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return image


if __name__ == '__main__':
    import argparse
    import cv2
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser('Display processed image.')
    parser.add_argument('-f', default='test4.jpg', help='name of test image')
    args = parser.parse_args()

    image = cv2.imread('test_images/{}'.format(args.f))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = process_image(image, debug=True)
