from collections import deque
from moviepy.editor import VideoFileClip

from camera_undistort import undistort_image
from threshold import combined_threshold
from perspective_transform import original2bird_eye
from find_lane_line import calc_lane_lines
from image_pipline import draw_lane_lines

prev_fit = False
curve_queue = deque(maxlen=7)


def process_video_frame(image):
    """process the frame and return the processed result"""
    global prev_fit

    undistorted = undistort_image(image)
    threshold = combined_threshold(undistorted)
    bird_eye = original2bird_eye(threshold)
    lane_line_params = calc_lane_lines(bird_eye, prev_fit=prev_fit)

    # Compute confidence by recent two curvature and decide how to search in next frame
    curve = lane_line_params['curvature']
    last_curve = curve_queue[0] if len(curve_queue) else curve
    confident = abs(curve - last_curve) / curve < 0.2
    if not confident:
        prev_fit = False
    else:
        prev_fit = lane_line_params['left_fit'], lane_line_params['right_fit']

    # maintain the latest curves and output an average to draw
    curve_queue.appendleft(curve)
    lane_line_params['curvature'] = sum(curve_queue) / len(curve_queue)

    result = draw_lane_lines(undistorted, lane_line_params, prev_fit=prev_fit)
    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Process video.')
    parser.add_argument('-f', default='test_videos/challenge_video.mp4', help='path to viedo')
    parser.add_argument('-o', default=None, help='path to output video')
    args = parser.parse_args()

    clip2 = VideoFileClip(args.f)
    vid_clip = clip2.fl_image(process_video_frame)

    output = args.o if args.o else args.f.replace('.', '_output.')
    vid_clip.write_videofile(output, audio=False)
