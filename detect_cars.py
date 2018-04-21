#!/usr/bin/python

#imports
from helper_functions import *

from moviepy.editor import VideoFileClip
from functools import partial
from scipy.ndimage.measurements import label

import time
import gc
from collections import deque

# ---------------------------------


DATA_LIMIT = 0
C_DEBUG = False
VID_DEBUG = False
WRITE_VIDEO = True
C_PERSISTANT_CLASSIFIER_DATA = True

_EXPORT_PICTURES_ = False

color_space = 'YUV'  # RGB/HSV/LUV/HLS/YUV
hog_orient = 10  #
hog_pix_per_cell = 12
hog_cell_per_block = 2
spatial_size = (32, 32)
hist_bins = 32
test_size = 0.2

heat_map_num_frames = 12
heat_map_thres = 3.5

y_start_stop = [400, None]  # Min and max in y to search in slide_window() # not used anymore
xy_window = (95,95) # not used anymore
xy_overlap = (0.8,0.8) # not used anymore

find_car_settings_list =[] # list of ystart, ystop, scale alternatives
find_car_settings_list.append((400,656,2.0))
find_car_settings_list.append((400,592,1.5))
find_car_settings_list.append((400,528,1.0))


# heatmap que
car_heatmap_buffer = deque(maxlen=heat_map_num_frames)

def process_frame(classifier,scaler,im):

    if C_DEBUG:
        show_image(im)


    #get_car_windows(classifier,scaler,im)

    # find cars in different window sizes
    windows = []
    for setting in find_car_settings_list:
        w = find_cars(im, setting[0], setting[1], setting[2], classifier, scaler, hog_orient, hog_pix_per_cell, hog_cell_per_block, spatial_size, hist_bins)
        windows += w

    # store last windows
    global car_heatmap_buffer
    if getResetAllways():
        car_heatmap_buffer = deque(maxlen=heat_map_num_frames)

    car_heatmap_buffer.append(windows)

    # create heatmap
    heatmap = np.zeros_like(im[:,:,0]).astype(np.float)

    for ws in car_heatmap_buffer:
        heatmap = add_heat(heatmap,ws)

    if C_DEBUG:
        heat_img = np.clip(heatmap, 0, 255)
        max=np.max(heat_img)
        heat_img=heat_img*(255/max)
        show_image(heat_img, True)

    if len(car_heatmap_buffer) > 1: # take a single frame as it is, apply thres otherwise
        heatmap = apply_threshold(heatmap,1+len(car_heatmap_buffer)*heat_map_thres)

    if C_DEBUG:
        # make images
        box_img = draw_boxes(im,windows)
        show_image(box_img)

        heat_img = np.clip(heatmap,0,255)
        show_image(heat_img,True)

    box_labels = label(heatmap)
    box_filtered_img = draw_labeled_bboxes(im, box_labels)

    if C_DEBUG:
        show_image(box_filtered_img)

    processed_img = box_filtered_img

    return processed_img

bFirst_images = True

def train_classifier(path_images_vehicle,path_images_non_vehicle,bPersistant=False):
    data_file = "classifier.dat"

    # prevent training each run, just load the data if available
    if bPersistant:
        print("Check for existing classifier data ...")
        try:
            (svc, X_scaler) = load_object(data_file)
            print("Loaded classifier data")
            return (svc, X_scaler)
        except:
            print("Could not load classifier data, training again ...")

    print("Loading images ...")
    images_vehicle = load_images_rec(path_images_vehicle, maxCount = DATA_LIMIT)
    print("Vehicles: " + str(len(images_vehicle)))
    images_non_vehicle = load_images_rec(path_images_non_vehicle, maxCount=DATA_LIMIT)
    print("Non-Vehicles: " + str(len(images_non_vehicle)))

    # get features
    # from/based on Lesson 0 #28
    print ('Extracting features ...')
    # spatial, color, hog1,2,3
    features_vehicle = extract_features(images_vehicle, cspace=color_space,
                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                        hog_orient=hog_orient, hog_pix_per_cell=hog_pix_per_cell,
                                        hog_cell_per_block=hog_cell_per_block)
    print ('Got vehicle features.')

    features_non_vehicle = extract_features(images_non_vehicle, cspace=color_space,
                                            spatial_size=spatial_size, hist_bins=hist_bins,
                                            hog_orient=hog_orient, hog_pix_per_cell=hog_pix_per_cell,
                                            hog_cell_per_block=hog_cell_per_block)
    print ('Got non-vehicle features.')

    if C_DEBUG:
        #get an image
        global bFirst_images
        if bFirst_images:
            print(images_vehicle[0].shape)
            idx_car = 6
            idx_non = 0
            img_car = get_hog_img_3_chan(convert_color(images_vehicle[idx_car],'RGB2YUV'),hog_orient,hog_pix_per_cell,hog_cell_per_block)
            img_non_car = get_hog_img_3_chan(convert_color(images_non_vehicle[idx_non],'RGB2YUV'), hog_orient, hog_pix_per_cell, hog_cell_per_block)
            show_image(images_vehicle[idx_car]*255)
            show_image(img_car*255)
            show_image(images_non_vehicle[idx_non]*255)
            show_image(img_non_car*255)


    X = np.vstack((features_vehicle, features_non_vehicle)).astype(np.float64)
    # Define the labels vector. 1: car 0: no car
    y = np.hstack((np.ones(len(features_vehicle)), np.zeros(len(features_non_vehicle))))
    #print(y.shape)

    # split test data
    rnd = np.random.randint(0, 100)
    print("Splitting data ...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rnd)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    print('Using spatial binning of:', spatial_size,
          'and', hist_bins, 'histogram bins')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    print("Training ...")
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    if bPersistant:
        save_object((svc, X_scaler),data_file)
        print("Saved classifier data")

    return (svc, X_scaler)

# not used anymore, using find_cars instead
def get_car_windows(classifier,scaler,img):

    search_img = np.copy(img).astype(np.float32)/255

    windows = slide_window(search_img,y_start_stop=y_start_stop,xy_window=xy_window,xy_overlap=xy_overlap)

    car_windows = search_windows(search_img,windows,classifier,scaler,color_space=color_space,spatial_size=spatial_size,
                                 hist_bins=hist_bins,orient=hog_orient,pix_per_cell=hog_pix_per_cell,
                                 cell_per_block=hog_cell_per_block)

    if C_DEBUG:
        box_img = np.copy(img)
        box_img = draw_boxes(box_img,car_windows)
        show_image(box_img)

    return car_windows



def main():
    # main, do something
    global C_DEBUG # may be changed here

    Video = WRITE_VIDEO

    if _EXPORT_PICTURES_:
        setWriteAllImages(True)
        C_DEBUG = True
        Video = False

    #C_DEBUG = True

    ## train
    #training data paths
    root_path_vehicles = 'training/vehicles'
    root_path_non_vehicles = 'training/non-vehicles'
    classifier, scaler = train_classifier(root_path_vehicles,root_path_non_vehicles,C_PERSISTANT_CLASSIFIER_DATA)

    # define partial function for arguments
    single_param_process_frame = partial(process_frame,classifier,scaler)

    ## Preparation Done ---- Looping images / video from here on
    if Video:
        print("Processing video ...")
        setResetAllways(False)
        setNoShow(True)
        project_video_output = "project_video_output.mp4"
        project_video_input = VideoFileClip("project_video.mp4")
        #project_video_input = VideoFileClip("test_video.mp4")

        processed_project_video = project_video_input.fl_image(single_param_process_frame)
        processed_project_video.write_videofile(project_video_output, audio=False)
        setNoShow(False)

    else:
        print("Processing single images ...")
        setResetAllways(True) # we need to reset means and medians as the images are independent
        images = load_images('test_images/')
        for im in images:
            res = single_param_process_frame(im)

    print("Done!")
    return 0


if __name__ == '__main__':
    main()