import numpy as np
import cv2

# source matrix
src = np.array([[585, 460],
                [203, 720],
                [1127, 720],
                [695, 460]], dtype=np.float32)

# target matrix to transform to
dst = np.array([[320, 0],
                [320, 720],
                [960, 720],
                [960, 0]], dtype=np.float32)

# transform matrix
trans_matrix = cv2.getPerspectiveTransform(src, dst)
invert_matrix = cv2.getPerspectiveTransform(dst, src)


def original2bird_eye(image):
    return cv2.warpPerspective(image, trans_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)


def bird_eye2original(image):
    return cv2.warpPerspective(image, invert_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser('Display bird eye transformed image.')
    parser.add_argument('-f', default='test1.jpg', help='name of test image')
    args = parser.parse_args()

    image = cv2.imread('test_images/{}'.format(args.f))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = original2bird_eye(image)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(result)
    ax2.set_title('Transformed Image', fontsize=30)
    plt.show()

