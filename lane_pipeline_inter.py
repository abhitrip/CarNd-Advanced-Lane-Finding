import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import collections
from LaneLine import Line
# For Displaying/Viewing/Editing video on Ipython console
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import pickle

# Step1. Camera Calibration

import os
calibration_pickle = './camera_cal/calib_data.pkl'
image_dir = 'camera_cal/calibration*.jpg'
def calibrateCamera():
    nx,ny = 9,6
    if os.path.isfile(calibration_pickle):
        with open(calibration_pickle,'rb') as pkl:
            retflags, mtx, dist, rvecs, tvecs = pickle.load(pkl)
            return retflags, mtx, dist, rvecs, tvecs

    # Prepare object points like (0,0,0), (1,0,0), (2,0,0), ... (8,5,0)
    objp = np.zeros((nx*ny,3),np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # Create Lists to store object points and image points from all the images
    objpoints = [] # 3d points of object in real world co-ordinates
    imgpoints = [] # 2d points of projected image in camera co-ordinates

    # Get the list of image files
    images = glob.glob(image_dir)

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #Find chessboard corners
        retflags, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If corner found, then add to object and image points
        if retflags == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx,ny), corners, retflags)
            plt.imshow(img)
            plt.show()

    retflags, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints, (gray.shape[1], gray.shape[0]), None, None)
    with open(calibration_pickle,'wb') as pkl_dump:
        pickle.dump((retflags, mtx, dist, rvecs, tvecs), pkl_dump)

    return retflags, mtx, dist, rvecs, tvecs


def undistort(img, mtx, dist):
    frame_undistorted = cv2.undistort(img, mtx, dist, newCameraMatrix=mtx)
    return frame_undistorted

# Step 1.1 Test Undistortion

ret, mtx, dist, rvecs, tvecs = calibrateCamera()
img = mpimg.imread('test_images/test2.jpg')
img_undistorted = undistort(img, mtx, dist)

"""
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(24,12))
fig.tight_layout()
ax1.imshow(img)
ax1.set_title('Distorted image', fontsize=20)
ax2.imshow(img_undistorted)
ax2.set_title('Undistorted image',fontsize=20)
plt.show()
"""

# Step 2. Processing Images and finding edges
def equializeThreshGray(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    eq_gray = cv2.equalizeHist(gray)
    _, thimg = cv2.threshold(eq_gray, thresh=250, maxval= 255, type=cv2.THRESH_BINARY)
    return thimg


def sobelImg(img, thresh_min = 25, thresh_max = 255, sobel_kernel = 11):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = np.absolute(cv2.Sobel(grayImg,cv2.CV_64F,1,0, ksize=sobel_kernel))
    sobely = np.absolute(cv2.Sobel(grayImg,cv2.CV_64F,0,1, ksize=sobel_kernel))
    scaled_sobelx = np.uint16(255*sobelx/np.max(sobelx))
    scaled_sobely = np.uint16(255*sobely/np.max(sobely))
    sobel_sum = scaled_sobelx+0.2*scaled_sobely
    scaled_sobel_sum = np.uint8(255*sobel_sum/np.max(sobel_sum))
    sum_binary = np.zeros_like(scaled_sobel_sum)
    sum_binary[(scaled_sobel_sum >= thresh_min) & (scaled_sobel_sum <= thresh_max)] = 1
    return sum_binary

def sobelMagnitude(img, thresh_min=75, thresh_max=255, sobel_kernel=11):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = np.absolute(cv2.Sobel(grayImg, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobely = np.absolute(cv2.Sobel(grayImg, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scaled_gradmag = np.uint8(255*gradmag/np.max(gradmag))
    gradmag_binary = np.zeros_like(scaled_gradmag)
    gradmag_binary[(scaled_gradmag >= thresh_min) & (scaled_gradmag <= thresh_max)] = 1
    return gradmag_binary


def hsvThres(img, thresh_min , thresh_max ):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_gt = np.all(hsv>thresh_min, axis=2)
    mask_lt = np.all(hsv<thresh_max, axis=2)
    hsv_mask = np.logical_and(mask_gt, mask_lt)
    return hsv_mask

# Now let's make the main image processing pipeline

def binariseImage(img, displayImages=True):
    h, w, c = img.shape
    hls_min, hls_max = ([0,70,70], [50, 255, 255])

    binary_mask = np.zeros((h, w), dtype = np.uint8)
    # First detect yellow lines
    s_mask = hsvThres(img, hls_min,hls_max)

    binary_mask = np.logical_or(binary_mask,s_mask)


    # now detect white lines by thresholding equialised frame

    white_mask = equializeThreshGray(img)
    binary_mask = np.logical_or(binary_mask, white_mask)

    # get sobel max
    sob_min, sob_max = 75, 255
    # Now do a red channel thresholding
    sob_mask = sobelMagnitude(img, thresh_min=75, thresh_max=255, sobel_kernel=9)
    binary_mask = np.logical_or(binary_mask, sob_mask)


    # Now do a simple dilate then erode to fill holes, to keep lines continuous
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    print("closing type = %s" %str(closing.dtype))
    if displayImages==True:
        fig, ax = plt.subplots(2, 3,figsize=(15,6))
        fig.tight_layout()
        ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        ax[0, 0].set_title('input_frame')

        ax[0, 0].set_axis_bgcolor('red')
        ax[0, 1].imshow(white_mask, cmap='gray')


        ax[0, 2].imshow(s_mask, cmap='gray')
        ax[0, 2].set_title('yellow mask')


        ax[1, 0].imshow(sob_mask, cmap='gray')
        ax[1, 0].set_title('sobel mask')


        ax[1, 1].imshow(binary_mask, cmap='gray')
        ax[1, 1].set_title('Final Binary mask')


        ax[1, 2].imshow(closing, cmap='gray')
        ax[1, 2].set_title('after filling')
        plt.show()
    return closing

"""
test_images = glob.glob('test_images/*.jpg')

img = cv2.imread("test_images/test2.jpg")
closed = binariseImage(img, True)
"""
# Now let's do the perspective transform part

def drawQuadrilateral(img, pts, colour = [255,0,0], thickness=4 ):
    pt1, pt2, pt3, pt4 = pts
    cv2.line(img, tuple(pt1), tuple(pt2), colour, thickness)
    cv2.line(img, tuple(pt2), tuple(pt3), colour, thickness)
    cv2.line(img, tuple(pt3), tuple(pt4), colour, thickness)
    cv2.line(img, tuple(pt4), tuple(pt1), colour, thickness)
    cv2.imshow("rectangledImg",cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    cv2.waitKey()

def perspectiveTransform(img, displayImages=True, showQuad=True):
    h,w = img.shape[:2]
    src_pts = np.float32([[0, h-30],[w,h-30],[540,450],[750,450]])
    dst_pts = np.float32([[0,h],[w,h],[0,0],[w,0]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts,src_pts)
    warped_img = cv2.warpPerspective(img, M, (w,h), flags = cv2.INTER_LINEAR)


    if displayImages==True:
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,9))
        fig.tight_layout()
        ax1.imshow(img, cmap='gray')
        ax1.set_title(" original image")
        ax2.imshow(warped_img, cmap='gray')
        ax2.set_title(" warped image- bird'seye view")
        plt.show()
    return warped_img, M, Minv

# show result on test images
# for test_img in glob.glob('test_images/*.jpg'):
"""
test_img = "test_images/test4.jpg"
img = mpimg.imread(test_img)
img_undistorted = undistort(img, mtx, dist)
img_binary = binariseImage(img_undistorted,displayImages=False)
img_birdeye, M, Minv = perspectiveTransform(img_binary, True)
print(img_binary.shape)
"""
## Now do the sliding window fitting part

# Define method to fit polynomial to binary image with lines extracted, using sliding window
ym_per_pix = 30 / 720   # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
time_window = 15        # results are averaged over this number of frames
def sliding_windows(birdeye_binary, line_lt, line_rt, n_windows=9, print_image=False):
    height, width = birdeye_binary.shape

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(birdeye_binary[height//2:-30, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = len(histogram) // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(height / n_windows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = birdeye_binary.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    margin = 100  # width of the windows +/- margin
    minpix = 50   # minimum number of pixels found to recenter window

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(n_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low)
                          & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low)
                           & (nonzero_x < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    line_lt.allx, line_lt.ally = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
    line_rt.allx, line_rt.ally = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]

    detected = True
    if not list(line_lt.allx) or not list(line_lt.ally):
        left_fit_pixel = line_lt.lastFitPixel
        left_fit_meter = line_lt.lastFitMetPerPixel
        detected = False
    else:
        left_fit_pixel = np.polyfit(line_lt.ally, line_lt.allx, 2)
        left_fit_meter = np.polyfit(line_lt.ally * ym_per_pix, line_lt.allx * xm_per_pix, 2)

    if not list(line_rt.allx) or not list(line_rt.ally):
        right_fit_pixel = line_rt.last_fit_pixel
        right_fit_meter = line_rt.last_fit_meter
        detected = False
    else:
        right_fit_pixel = np.polyfit(line_rt.ally, line_rt.allx, 2)
        right_fit_meter = np.polyfit(line_rt.ally * ym_per_pix, line_rt.allx * xm_per_pix, 2)

    line_lt.update_line_eq(left_fit_pixel, left_fit_meter, detected=detected)
    line_rt.update_line_eq(right_fit_pixel, right_fit_meter, detected=detected)

    # Generate x and y values for plotting
    ploty = np.linspace(0, height - 1, height)
    left_fitx = left_fit_pixel[0] * ploty ** 2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
    right_fitx = right_fit_pixel[0] * ploty ** 2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]

    out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
    out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

    if print_image:
        f, ax = plt.subplots(1, 2, figsize=(15,6),  subplot_kw={'xticks': [], 'yticks': []})
        f.set_facecolor('white')
        ax[0].imshow(birdeye_binary, cmap='gray')
        ax[1].imshow(out_img)
        ax[1].plot(left_fitx, ploty, color='yellow')
        ax[1].plot(right_fitx, ploty, color='yellow')
        ax[1].set_xlim(0, 1280)
        ax[1].set_ylim(720, 0)

        plt.show()

    return line_lt, line_rt, out_img

# visualize the result on example image
def approx_by_previous_fits(birdeye_binary, line_lt, line_rt, print_image=False):
    """
    This function uses previously detected lane-lines to
    search of lane-lines in the current frame.
    """

    height, width = birdeye_binary.shape

    left_fit_pixel = line_lt.lastFitPixel
    right_fit_pixel = line_rt.lastFitPixel

    nonzero = birdeye_binary.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    margin = 100
    left_lane_inds = (
    (nonzero_x > (left_fit_pixel[0] * (nonzero_y ** 2) + left_fit_pixel[1] * nonzero_y + left_fit_pixel[2] - margin)) & (
    nonzero_x < (left_fit_pixel[0] * (nonzero_y ** 2) + left_fit_pixel[1] * nonzero_y + left_fit_pixel[2] + margin)))
    right_lane_inds = (
    (nonzero_x > (right_fit_pixel[0] * (nonzero_y ** 2) + right_fit_pixel[1] * nonzero_y + right_fit_pixel[2] - margin)) & (
    nonzero_x < (right_fit_pixel[0] * (nonzero_y ** 2) + right_fit_pixel[1] * nonzero_y + right_fit_pixel[2] + margin)))

    # Extract left and right line pixel positions
    line_lt.allx, line_lt.ally = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
    line_rt.allx, line_rt.ally = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]

    detected = True
    if not list(line_lt.allx) or not list(line_lt.ally):
        left_fit_pixel = line_lt.lastFitPixel
        left_fit_meter = line_lt.lastFitMetPerPixel
        detected = False
    else:
        left_fit_pixel = np.polyfit(line_lt.ally, line_lt.allx, 2)
        left_fit_meter = np.polyfit(line_lt.ally * ym_per_pix, line_lt.allx * xm_per_pix, 2)

    if not list(line_rt.allx) or not list(line_rt.ally):
        right_fit_pixel = line_rt.lastFitPixel
        right_fit_meter = line_rt.lastFitMetPerPixel
        detected = False
    else:
        right_fit_pixel = np.polyfit(line_rt.ally, line_rt.allx, 2)
        right_fit_meter = np.polyfit(line_rt.ally * ym_per_pix, line_rt.allx * xm_per_pix, 2)

    line_lt.update_line_eq(left_fit_pixel, left_fit_meter, detected=detected)
    line_rt.update_line_eq(right_fit_pixel, right_fit_meter, detected=detected)

    # Generate x and y values for plotting
    ploty = np.linspace(0, height - 1, height)
    left_fitx = left_fit_pixel[0] * ploty ** 2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
    right_fitx = right_fit_pixel[0] * ploty ** 2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]

    # Create an image to draw on and an image to show the selection window
    img_fit = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary)) * 255
    window_img = np.zeros_like(img_fit)

    # Color in left and right line pixels
    img_fit[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
    img_fit[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(img_fit, 1, window_img, 0.3, 0)

    if print_image:
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

        plt.show()

    return line_lt, line_rt, img_fit

line_lt, line_rt = Line(frameBuffLen=10), Line(frameBuffLen=10)
img = cv2.imread("test_images/test5.jpg")
img_undistorted = undistort(img, mtx, dist)
img_binary = binariseImage(img_undistorted)
img_birdeye, M, Minv = perspectiveTransform(img_binary)
line_lt, line_rt, img_out = sliding_windows(img_birdeye, line_lt, line_rt, n_windows=7, print_image=True)

def combine_line_with_road(img_undistorted, Minv, leftLine, rightLine, keepCurr):
    """
    Combine into 1 image, the driving boundary, lane lines and original image
    """
    height,width = img_undistorted.shape[:2]
    left_fit = leftLine.avgFit if keepCurr else leftLine.lastFitPixel
    right_fit = rightLine.avgFit if keepCurr else rightLine.lastFitPixel

    # Now get x and y co-ords for plotting
    ploty = np.linspace(0, height-1, height)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # draw road as green polygon on original frame
    road_warp = np.zeros_like(img_undistorted, dtype=np.uint8)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(road_warp, np.int_([pts]), (0, 255, 0))
    road_dewarped = cv2.warpPerspective(road_warp, Minv, (width, height))  # Warp back to original image space

    combined_img = cv2.addWeighted(img_undistorted, 1., road_dewarped, 0.3, 0)

    # now separately draw solid lines to highlight them
    line_warp = np.zeros_like(img_undistorted)
    line_warp = line_lt.draw_on_img(line_warp, colour=(255, 0, 0), avg=keepCurr)
    line_warp = line_rt.draw_on_img(line_warp, colour=(255, 0, 0), avg=keepCurr)
    line_dewarped = cv2.warpPerspective(line_warp, Minv, (width, height))

    lines_mask = combined_img.copy()
    idx = np.any([line_dewarped != 0][0], axis=2)
    lines_mask[idx] = line_dewarped[idx]

    combined_img = cv2.addWeighted(src1=lines_mask, alpha=0.8, src2=combined_img, beta=0.5, gamma=0.)

    return combined_img

def add_metrics_to_img(combined_img, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter):
    """
   Print the final image with radius of curvature.
    """
    h, w = combined_img.shape[:2]

    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15

    # add a gray rectangle to highlight the upper area
    mask = combined_img.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(500, thumb_h+2*off_y), colourr=(0, 0, 0), thickness=cv2.FILLED)
    combined_img = cv2.addWeighted(src1=mask, alpha=0.2, src2=combined_img, beta=0.8, gamma=0)

    # add text (curvature and offset info) on the upper right of the blend
    mean_curvature_meter = np.mean([line_lt.curvatureMetPerPixel, line_rt.curvatureMetPerPixel])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined_img, 'Radius of curvature: {:.02f}m'.format(mean_curvature_meter), (30, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined_img, 'Offset from center: {:.02f}m'.format(offset_meter), (30, 130), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return combined_img

def get_offset_from_center(line_lt, line_rt, frame_width):
    """
    Compute offset from center of the inferred lane.
    The offset from the lane center can be computed under the hypothesis that the camera is fixed
    and mounted in the midpoint of the car roof. In this case, we can approximate the car's deviation
    from the lane center as the distance between the center of the image and the midpoint at the bottom
    of the image of the two lane-lines detected.
    """
    if line_lt.detected and line_rt.detected:
        line_lt_bottom = np.mean(line_lt.allx[line_lt.ally > 0.95 * line_lt.ally.max()])
        line_rt_bottom = np.mean(line_rt.allx[line_rt.ally > 0.95 * line_rt.ally.max()])
        lane_width = line_rt_bottom - line_lt_bottom
        midpoint = frame_width / 2
        offset_pix = abs((line_lt_bottom + lane_width / 2) - midpoint)
        offset_meter = xm_per_pix * offset_pix
    else:
        offset_meter = -1

    return offset_meter

def process_pipeline(frame, keepCurr=True):
    """
    Apply the whole LD pipeline to a single frame
    """
    global leftLine,rightLine,processedFrames
    # First remove distortion of image frame
    img_undistorted = undistort(frame,mtx,dist)
    # Now binarise image
    img_binary = binariseImage(img,False)
    # Now Compute Perspective Transform
    img_birdeye, M, Minv = perspectiveTransform(img_binary)
    # fit 2-degree polynomial curve onto lane lines found
    if processedFrames>0 and keepCurr and leftLine.detected and rightLine.detected:
        leftLine, rightLine, img_fit = approx_by_previous_fits(img_birdeye, leftLine, rightLine)
    else:
        leftLine, rightLine, img_fit = sliding_windows(img_birdeye, leftLine, rightLine)

    # Compute offset in meter from ctr of frame
    offset = get_offset_from_center(leftLine,rightLine,frame_width=frame.shape[1])

    # draw the surface enclosed by lane lines back onto the original frame
    combined_img = combine_line_with_road(img_undistorted, Minv, leftLine, rightLine, keepCurr)

    # stitch on the top of final output images from different steps of the pipeline
    blend_output = add_metrics_to_img(combined_img, img_binary, img_birdeye, img_fit, leftLine, rightLine, offset)

    processedFrames+=1
    return cv2.cvtColor(blend_output, cv2.COLOR_BGR2RGB)

images = glob.glob(os.path.join("test_images", '*.jpg'))
"""
for filename in images:
    img = mpimg.imread(filename)
    blend = process_pipeline(img, keepCurr=False)
    cv2.imwrite("output_" + filename, blend)
"""
processedFrames = 0
leftLine, rightLine = Line(frameBuffLen=10), Line(frameBuffLen=10)
frame = cv2.imread("test_images/test5.jpg")
blend = process_pipeline(frame, keepCurr=False)





