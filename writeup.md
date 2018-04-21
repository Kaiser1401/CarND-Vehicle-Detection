## Writeup 

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References) 
[im1]: ./output_images/i_0.png ""
[im2]: ./output_images/i_2.png ""
[hog_1]: ./output_images/i_1.png ""
[hog_2]: ./output_images/i_3.png ""
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

## Code Files
The code for the project is written in two files:

* `detect_cars.py` Entry main function, image/frame processing
* `helper_functions.py`All the functions needed for that


## Training

In `train_classifier()` in `detect_cars.py:~105`, if there is no saved classifier data, first, all vehicle and non-vehicle images are loaded and several features for each image are extracted with `extract_features()` (`helper_functions.py:~237`) which is based on the lessons code as well.

| Example vehicle | Example non-vehicle |
|:--:|:--:|
|![][im1]|![][im2]|

All training / processing is based on YUV-colorspace images which represents images as a full black and white layer plus two additional color information channels. Different color spaces have been tried but YUV seems to give reasonable results.

The used features are:

#### 1 Spatial Color binning

One feature element used for training and classification is spatial binning of the seperate color channels with a binning size of 32x32px (`helper_functions.py:~230`)

#### 2 Color histogram

Another element used ist the histogram of the single channels in the images. These are grouped into 32 bins each.  (`helper_functions.py:~218`)


#### 3 Histogram of Oriented Gradients (HOG)

HOG feature extraction is based on the `get_hog_features()` from lesson 0 # 35 and is implemented in `helper_functions.py:~196`. Several numbers of orientation, cell and block sizes, color channels and combinations thereof have been tested. I found the following set to give a good result:

|param|value|
|--|--|
|channels| all from YUV image |
|orientations|10|
|pixel per cell | 12|
|cells per block|2|

The following images show the visualized HOG features for each channel

|| vehicle | non-vehicle |
|--|:--:|:--:|
|input|![][im1]|![][im2]|
|HOG(YUV)|![][hog_1]|![][hog_2]|

### Classifier

Combining the above mentioned features into single vectors (`extract_features()`) I trained a linear Support Vector Machine after randomizing and splitting the testdata. (`detect_cars.py:~179`). With the features used, I got a test accuracy of  0.9899.


## Processing Pipeline

In the beginnig I used zwo seperate implementations to process single frames and videos, one based on the seperate `slide_window()`-`search_windows()` pipeline presented in the lessen and one based on the `find_cars()` implementation from lesson0 #35.

In the final implementation only the latter is used. For video and single (example) image processing alike. Single images don't combine informations from last frame though for obvious reasons.

Processing starts in `process_frame()` (`detect_cars.py:~49`)

The steps for each picture / frame are as follows:

[i4]: ./output_images/i_4.png ""
[i5]: ./output_images/i_5.png ""
[i6]: ./output_images/i_6.png ""
[i8]: ./output_images/i_8.png ""

[i9]: ./output_images/i_9.png ""
[i10]: ./output_images/i_10.png ""
[i11]: ./output_images/i_11.png ""
[i13]: ./output_images/i_13.png ""

[i19]: ./output_images/i_19.png ""
[i20]: ./output_images/i_20.png ""
[i21]: ./output_images/i_21.png ""
[i23]: ./output_images/i_23.png ""

| Step | Ex1 | Ex2 | Ex3 |
|:--:|:--:|:--:|:--:|
| 0 Input |![][i4]|![][i9]|![][i19]|
| 1 Window Search|![][i6]|![][i11]|![][i21]|
| 2 Heatmap |![][i5]|![][i10]|![][i20]|
| 3 Combining  |![][i8]|![][i13]|![][i23]|

#### 1 Sliding Window Search

In `find_cars()` (`helper_functions.py:~362`) the input frame is walked through with overlapping windows of size 32, 48 and 64 pixels in the lower part (y > 400px) of the image. These windows hog, spatial and color histogram features are then fed to the classifier to decide weather there is a car in this window or not. If so, the mathcing windows are returned as bounding boxes.

#### 2 Heatmap

To remove outliers over several frames (and to combine the boxes), a heatmap is generated at the positions of the optained bounding boxes over the last 12 frames  (`detect_cars.py:~73`).

#### 3 Thresholding and Combining

To filter outliers, a threshold is used (`detect_cars.py:~83`) on the heatmap. The remaining part of the heatmp is used to label each region and to generate a combined bounding box for each (`detect_cars.py:~93`). These boxes are drawn on the frame as the result. 

## Video

The output video can be found here:  [project_video_output.mp4](project_video_output.mp4).

## Discussion
The implemented pipeline works reasonably on a video file. At about 2 frames/second on my laptop though this is far from realtime and needs a lot of processing improvement for real use. The usage of a threshold for outliers over several frames, which I tuned to get no false positives, also caused to 'loose' cars again relatively quickly. A threshold is nearly always a tradeof. Furthermore, taking into account several frames can introduce lag over that period. A more advanced way could be some kind of particle filter or other model based tracking methods.

