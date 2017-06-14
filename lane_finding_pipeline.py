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
"""
ret, mtx, dist, rvecs, tvecs = calibrateCamera()
img = mpimg.imread('test_images/test2.jpg')
img_undistorted = undistort(img, mtx, dist)
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(24,12))
fig.tight_layout()
ax1.imshow(img)
ax1.set_title('Distorted image', fontsize=20)
ax2.imshow(img_undistorted)
ax2.set_title('Undistorted image',fontsize=20)
plt.show()
"""
# Step 2. Processing Images and finding edges

def equalizeHist(img):
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img[:,:,1] = cv2.equalizeHist(img[:,:,1])
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])
    return img

def equializeThreshGray(img):
    eq_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
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

# Binary red channel threshold
def redThres(img, thresh_min = 25, thresh_max = 255):
    red = img[:,:,2]
    red_binary = np.zeros_like(red)
    red_binary[(red >= thresh_min) & (red <= thresh_max)]  = 1
    return red_binary

def sThres(img, thresh_min = 25, thresh_max = 255):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > thresh_min) & (s_channel <= thresh_max)] = 1
    return s_binary

# Return saturation channel
def getSatChannel(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    return hls[:,:,2]
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
"""
def binariseImage(img, displayImages=True):
    s_min, s_max = 70, 255
    eq_img = equalizeHist(img) # Equalized image
    r_min, r_max = 250, 255
    sob_min, sob_max = 75, 255
    # Now do a red channel thresholding
    h, w, c = img.shape
    binary_mask = np.zeros((h, w), dtype = np.uint8)

    # First do red threshold : For the white lines
    red_mask = redThres(eq_img, r_min, r_max)
    binary_mask = np.logical_or(red_mask, binary_mask)
    # do the sobel filter on the image
    sobel_mask = sobelMagnitude(eq_img, thresh_min=sob_min, thresh_max=sob_max, sobel_kernel=9)
    binary_mask = np.logical_or(sobel_mask, binary_mask)

    #now find the yellow line
    s_mask = sThres(eq_img, s_min, s_max)
    binary_mask = np.logical_or(s_mask, binary_mask)
    print("binary_mask = %s" %str(binary_mask.dtype))
    # Now do a simple dilate then erode to fill holes, to keep lines continuous
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
"""
def binariseImage(img, displayImages=True):
    h, w, c = img.shape
    s_min, s_max = 70, 255

    binary_mask = np.zeros((h, w), dtype = np.uint8)
    # First detect yellow lines
    s_mask = sThres(img, thresh_min=s_min, thresh_max=s_max)
    binary_mask = np.logical_or(binary_mask,s_mask)

    # now detect white lines by thresholding equialised frame
    eq_img = equalizeHist(img) # Equalized image
    white_mask = equializeThreshGray(eq_img)
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
        fig, ax = plt.subplots(2,3,figsize=(24,9))
        fig.tight_layout()
        ax[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0,0].set_title("original input frame")
        ax[0,1].imshow(white_mask,cmap='gray')
        ax[0,1].set_title("White Thresholed mask")
        ax[0,2].imshow(s_mask,cmap='binary')
        ax[0,2].set_title("Yellow mask")
        ax[1,0].imshow(sob_mask,cmap='gray')
        ax[1,0].set_title("sobel mask")
        ax[1,1].imshow(binary_mask,cmap='gray')
        ax[1,1].set_title("Final Mask from all channels")
        ax[1,2].imshow(closing,cmap='gray')
        ax[1,2].set_title("Mask with closed holes")
        plt.show()
    return closing
def HSV_select(image, min_values, max_values):
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    min_th = np.all(HSV > min_values, axis=2)
    max_th = np.all(HSV < max_values, axis=2)
    out = np.logical_and(min_th, max_th)
    return out


def sobel_select(image, kernel_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)
    _, sobel_mag = cv2.threshold(sobel_mag, 75, 1, cv2.THRESH_BINARY)
    return sobel_mag.astype(bool)


def equalized_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eq_global = cv2.equalizeHist(gray)
    _, th = cv2.threshold(eq_global, thresh=250, maxval=255, type=cv2.THRESH_BINARY)
    return th


def binarize(img, print_images=False):
    # threshold that works well for detecting yellow lanes
    HSV_min = np.array([0, 70, 70])
    HSV_max = np.array([50, 255, 255])

    h, w = img.shape[:2]

    binary = np.zeros(shape=(h, w), dtype=np.uint8)

    # extract yallow in HSV color space
    HSV_mask = HSV_select(img, HSV_min, HSV_max)
    binary = np.logical_or(binary, HSV_mask)

    # highlight white lines by thresholding the equalized frame
    white_mask = equalized_grayscale(img)
    binary = np.logical_or(binary, white_mask)

    # get Sobel mask
    sobel_mask = sobel_select(img, kernel_size=9)
    binary = np.logical_or(binary, sobel_mask)

        # apply a light morphology to "fill the gaps" in the binary image
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    if print_images:
#         plt.figure(figsize=(10,10))
        f, ax = plt.subplots(2, 3,figsize=(15,6))
        f.set_facecolor('white')
        ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0, 0].set_title('input_frame')
        #ax[0, 0].set_axis_off()
        ax[0, 0].set_axis_bgcolor('red')
        ax[0, 1].imshow(white_mask, cmap='gray')
        ax[0, 1].set_title('white mask')
        ax[0, 1].set_axis_off()

        ax[0, 2].imshow(HSV_mask, cmap='gray')
        ax[0, 2].set_title('yellow mask')
        #ax[0, 2].set_axis_off()

        ax[1, 0].imshow(sobel_mask, cmap='gray')
        ax[1, 0].set_title('sobel mask')
        #ax[1, 0].set_axis_off()

        ax[1, 1].imshow(binary, cmap='gray')
        ax[1, 1].set_title('Final Binary')
        #ax[1, 1].set_axis_off()

        ax[1, 2].imshow(closing, cmap='gray')
        ax[1, 2].set_title('after closure')
        #ax[1, 2].set_axis_off()
        plt.show()

    return closing


test_images = glob.glob('test_images/*.jpg')

img = cv2.imread("test_images/test2.jpg")
closed = binariseImage(img, True)













