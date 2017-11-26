import numpy as np
import cv2
from moviepy.editor import VideoFileClip

from camera_undistort import undistort_image
from threshold import combined_threshold
from perspective_transform import original2bird_eye
from find_lane_line import calc_lane_lines
from image_pipline import draw_lane_lines

prev_fit = None


def process_video_frame(image):
    global prev_fit

    undistorted = undistort_image(image)
    threshold = combined_threshold(undistorted)
    bird_eye = original2bird_eye(threshold)
    lane_line_params = calc_lane_lines(bird_eye, prev_fit=prev_fit)
    result = draw_lane_lines(undistorted, lane_line_params)

    prev_fit = lane_line_params['left_fit'], lane_line_params['right_fit']

    return result


if __name__ == '__main__':
    clip2 = VideoFileClip('test_videos/project_video.mp4')
    vid_clip = clip2.fl_image(process_video_frame)
    vid_clip.write_videofile('ss2_project_video.mp4', audio=False)
