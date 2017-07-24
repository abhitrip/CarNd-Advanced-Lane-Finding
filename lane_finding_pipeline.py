import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import collections

# For Displaying/Viewing/Editing video on Ipython console
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import pickle


## Step1. Camera Calibation

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
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    mask_gt = np.all(hls>thresh_min, axis=2)
    mask_lt = np.all(hls<thresh_max, axis=2)
    hsv_mask = np.logical_and(mask_gt, mask_lt)
    return hsv_mask

# Visualize an example
"""
img = cv2.imread('test_images/test2.jpg')
img = equalizeHist(img)
img_proc = img[:,:,2]+0.5*getSatChannel(img)
img_proc = np.uint8(255*img_proc/np.max(img_proc))
plt.imshow(img_proc, cmap='gray')
plt.show()
"""
# Now make the main image processing pipeline


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
# Perspective Transform

def drawQuadrilateral(img, pts, colour = [255,0,0], thickness=4 ):
    pt1, pt2, pt3, pt4 = pts
    cv2.line(img, tuple(pt1), tuple(pt2), colour, thickness)
    cv2.line(img, tuple(pt2), tuple(pt3), colour, thickness)
    cv2.line(img, tuple(pt3), tuple(pt4), colour, thickness)
    cv2.line(img, tuple(pt4), tuple(pt1), colour, thickness)


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
        for pt in src_pts:
            ax1.plot(*pt,'.')
        ax2.imshow(warped_img, cmap='gray')
        ax2.set_title(" warped image- bird'seye view")
        for pt in dst_pts:
            ax2.plot(*pt,'.')
        plt.show()
    return warped_img, M, Minv


# show result on test images
# for test_img in glob.glob('test_images/*.jpg'):
test_img = "test_images/test4.jpg"
img = cv2.imread(test_img)
img_undistorted = undistort(img, mtx, dist)
img_binary = binariseImage(img_undistorted,False)
img_birdeye, M, Minv = perspectiveTransform(img_undistorted, True)



## Now do the sliding window fitting part

