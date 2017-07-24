

# **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png 
[image2]: ./test_images/test2_transformed.jpg 
[image3]: ./test_images/test2_bin.jpg 
[image4]: ./test4_persp.jpg 
[image5]: ./test_images/sliding.jpg 
[image6]: ./test_images/road_detected.jpg 
[video1]: ./project_video_out.mp4 

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  



### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./advLaneFinding.ipynb" (Step 1. Camera Calibration headlined cell).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. To detect white lines , histogram equalisation is enough. However, inorder to detect yellow lines, I used the v-channel of image in hsv colour space. Finally, a sobel mask is used to compute the gradients. Also, I used 
morphological dilation and erosion to make the edge lines continuous. The code is shown in code cell - **2. Image Image preprocessing, thresholding and binarisation**


![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspectiveTransform()`, which appears in the `Perspective transform` cell  of the IPython notebook.  The `perspectiveTransform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[0, img_size[0] -30],
    [img_size[1] ,img_size[0] -30] ],
    [540, 450],
    [750,450])
dst = np.float32(
    [[0,img_size[0]],
    [img_size[1], img_size[0]],
    [0, 0],
    [img_size[1], 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 0, 690        | 0, 720        | 
| 1280, 690     | 1280, 720     |
| 540, 450      |  0, 0         |
| 1280, 450     | 1280, 0       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the same algorithm as mentioned in the video lectures. First, check the histogram of the lower half of the image and find the two peaks for the left and right lines. Then we use the sliding window method to work our way upwards and find the relevant points in the image which mark the lane. Next, Use the np.polyfit() method to fit a second degree polynomial to these points.  You can see the code in sliding_windows() function.
For videos, we can reuse the previously detected lane lines(of previous n frames) to get an approximation of lane line in current frame. You can see the code in  approx_by_previous_fits().

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The offset from the lane center can be computed assuming that the camera is fixed and mounted in the midpoint of the car roof. In this case, we can approximate the car's deviation from the lane center as the distance between the center of the image and the midpoint at the bottom of the image of the two lane-lines detected. This is done using the method compute_offset_from_center().
The radius of curvature can be computed using the coefficients of 2nd degree polynomial fitted to the lane lines. The code is in the method curvature_meter().



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The methods combine_with_road() and add_metrics_to_img() are used to draw the lane back onto the road.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust ?

The main challenge was to select optimum parameters for the gradient thresholds. It is hard, since thresholds were chosen based on the 6 test images, but they might not work for all images. In other words, a generic threshold parameter setting is almost impossible. 
Moreover, the pipeline is still not full proof to changes in brightness, contrast etc. 

To make the pipeline more robust:-
1. Use Dynamic thersholding i.e. considering separate threshold parameters for different horizontal sections of the image. 
2. Modifying our algorithm to use  condifence levels for fits. It means we can reject fits, whose confidence is below a threshold. For example, If either of left or right fit is not meeting the confidence, then we could use the previous fit.

