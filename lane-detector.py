import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle
import cv2
import glob
import time
import os
import argparse
from moviepy.editor import VideoFileClip

# CHANGE THESE
input_directory = '/media/max/Storage/comma-dataset/comma-dataset/images/'
output_directory = '/media/max/Storage/comma-dataset/comma-dataset/output/'

parser = argparse.ArgumentParser(description='Lane line detector')
parser.add_argument('--export_binary', dest='export_binary', action='store_true')
parser.set_defaults(export_binary=False)
args = parser.parse_args()

frame_begin = 10000
frame_end = 16000

def corners_unwarp(img):
    img_size = (img.shape[1], img.shape[0])

    src = np.float32([(110,40),
                  (210,40),
                  (20,120),
                  (300,120)])

    offset = 40
    dst = np.float32([(offset,0),
                  (img_size[0]-offset,0),
                  (offset,img_size[1]),
                  (img_size[0]-offset,img_size[1])])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M, Minv


def color_threshold(img):
    thresh = (180, 255)
    l_thresh = (190, 255)
    b_thresh = (190, 255)

    img_copy = np.copy(img)

    gray = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HLS).astype(np.float)
    lab = cv2.cvtColor(img_copy, cv2.COLOR_RGB2Lab)

    l_channel = hls[:,:,1]
    lab_b = lab[:,:,2]

    binary = np.zeros_like(gray)
    binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 50
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel > l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

    # Normalize if there is yellow in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    b_binary = np.zeros_like(lab_b)
    b_binary[(lab_b > b_thresh[0]) & (lab_b <= b_thresh[1])] = 1

    binary_warped = np.zeros_like(b_binary)
    binary_warped[(b_binary == 1) | (l_binary == 1) | (binary == 1) | (sxbinary == 1)] = 1

    return binary_warped

def preprocess(img):
    #undistorted = cal_undistort(img, objpoints, imgpoints)

    top_down, M, Minv = corners_unwarp(img)

    binary_warped = color_threshold(top_down)

    return binary_warped, Minv


def polyfit(binary_warped, visualize=False):
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 8
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 40
    # Set minimum number of pixels found to recenter window
    minpix = 20
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 1)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 1)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]

    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit, right_fit = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)


    if visualize:
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 320)
        plt.ylim(160, 0)

    return left_fit, right_fit, left_lane_inds, right_lane_inds

def polyfit_prev(binary_warped, left_fit_prev, right_fit_prev, visualize=False):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 20
    left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy +
    left_fit_prev[2] - margin)) & (nonzerox < (left_fit_prev[0]*(nonzeroy**2) +
    left_fit_prev[1]*nonzeroy + left_fit_prev[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy +
    right_fit_prev[2] - margin)) & (nonzerox < (right_fit_prev[0]*(nonzeroy**2) +
    right_fit_prev[1]*nonzeroy + right_fit_prev[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit, right_fit = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)

    if visualize:
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                      ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                                      ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 320)
        plt.ylim(160, 0)

    return left_fit, right_fit, left_lane_inds, right_lane_inds

def draw_on_lane(img, binary_warped, left_fit, right_fit, Minv):
    new_img = np.copy(img)
    if left_fit is None or right_fit is None:
        return original_img

    img_size = (binary_warped.shape[1], binary_warped.shape[0])

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, img_size[1]-1, num=img_size[1])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    #cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(0,255,0), thickness=5)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,0), thickness=5)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img_size[0], img_size[1]))
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    return result

def calculate_radius_and_distance(binary_warped, left_fit, right_fit, left_lane_inds, right_lane_inds):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    left_rad = 0
    right_rad = 0
    distance = 0

    img_size = (binary_warped.shape[1], binary_warped.shape[0])
    ploty = np.linspace(0, img_size[1]-1, img_size[1])
    y_eval = np.max(ploty)

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(leftx) > 0 and len(rightx) > 0:
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

        left_rad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_rad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    # Calculate how far the car is from the center of the lane
    if left_fit is not None and right_fit is not None:
        position = img_size[0] / 2
        left_x_intercept = left_fit[0]*img_size[1]**2 + left_fit[1]*img_size[1] + left_fit[2]
        right_x_intercept = right_fit[0]*img_size[1]**2 + right_fit[1]*img_size[1] + right_fit[2]
        lane_center = (left_x_intercept + right_x_intercept) / 2

        distance = (position - lane_center) * xm_per_pix

    return left_rad, right_rad, distance

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.recent_fits = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')

    def add_fit(self, fit, inds):
        if fit is not None:
            if len(self.recent_fits) > 0:
                self.diffs = np.absolute(self.recent_fits[len(self.recent_fits)-1] - fit)

            if (self.diffs[0] > 0.01 or self.diffs[1] > 1.0 or self.diffs[2] > 100.) and len(self.recent_fits) > 0:
                self.detected = False
            else:
                self.detected = True
                self.px_count = np.count_nonzero(inds)
                self.recent_fits.append(fit)
                if len(self.recent_fits) > 5:
                    # Remove oldest fit
                    self.recent_fits.pop(0)
                self.best_fit = np.average(self.recent_fits, axis=0)
        else:
            self.detected = False

            if len(self.recent_fits) > 0:
                # if there are still any fits in the queue, best_fit is their average
                self.best_fit = np.average(self.recent_fits, axis=0)


# In[28]:


def draw_data(img, radius, distance):
    new_img = np.copy(img)
    h = new_img.shape[0]
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Curve radius: ' + '{:04.2f}'.format(radius) + 'm'
    cv2.putText(new_img, text, (10,20), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
    direction = ''
    if distance > 0:
        direction = 'right'
    elif distance < 0:
        direction = 'left'
    abs_distance = abs(distance)
    text = '{:04.3f}'.format(abs_distance) + 'm ' + direction + ' of center'
    cv2.putText(new_img, text, (10,40), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return new_img


# In[32]:


def process_image(img):
    new_img = np.copy(img)
    binary_warped, Minv = preprocess(new_img)

    #if not left_line.detected or not right_line.detected:
    #    left_fit, right_fit, left_lane_inds, right_lane_inds = polyfit(binary_warped)
    #else:
    #    left_fit, right_fit, left_lane_inds, right_lane_inds = polyfit_prev(binary_warped, left_line.best_fit, right_line.best_fit)

    left_fit, right_fit, left_lane_inds, right_lane_inds = polyfit(binary_warped)

    left_line.add_fit(left_fit, left_lane_inds)
    right_line.add_fit(right_fit, right_lane_inds)

    if left_line.best_fit is not None and right_line.best_fit is not None:
        left_rad, right_rad, distance = calculate_radius_and_distance(binary_warped, left_line.best_fit, right_line.best_fit, left_lane_inds, right_lane_inds)

        if (args.export_binary):
            output_img = draw_on_lane(np.zeros_like(new_img), binary_warped, left_line.best_fit, right_line.best_fit, Minv)
        else:
            output_img = draw_on_lane(new_img, binary_warped, left_line.best_fit, right_line.best_fit, Minv)
    else:
        output_img = new_img

    return output_img

left_line = Line()
right_line = Line()

for i in range(frame_begin, frame_end):
    img = cv2.imread(input_directory + str(i) + '.jpg')

    if img is not None:

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output = process_image(rgb)
        bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        # Uncomment this to show the frames being output
        #cv2.imshow('frame', bgr)
        cv2.imwrite(output_directory + str(i) + '.jpg', bgr)

        if cv2.waitKey(25) == 27:
            break
