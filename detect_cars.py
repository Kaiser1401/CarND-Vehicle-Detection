#!/usr/bin/python

#imports
from helper_functions import *

from moviepy.editor import VideoFileClip
from functools import partial

import time
import gc

# ---------------------------------


DATA_LIMIT = 10
C_DEBUG = True
VID_DEBUG = False
WRITE_VIDEO = False
C_PERSISTANT_CLASSIFIER_DATA = False

_EXPORT_PICTURES_ = False

color_space = 'YUV'  # RGB/HSV/LUV/HLS/YUV
hog_orient = 10  #
hog_pix_per_cell = 12
hog_cell_per_block = 2
spatial_size = (32, 32)
hist_bins = 32
test_size = 0.2

y_start_stop = [400, None]  # Min and max in y to search in slide_window()

def process_frame(im):

    if C_DEBUG:
        show_image(im)


#    if C_DEBUG:
#        plt.plot(hist)
#        show_image(fig2data(plt.gcf()))
#        plt.clf()

    processed_img = im

    return processed_img


def train_classifier(images_vehicle,images_non_vehicle,bPersistant=False):
    data_file = "classifier.dat"

    if bPersistant:
        try:
            classifier = load_object(data_file)
            print("Loaded classifier data")
            return classifier
        except:
            print("Could not load classifier data, training again ...")





    # get features
    # from/based on Lesson 0 #28
    print ('Extracting features ...')
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

    X = np.vstack((features_vehicle, features_non_vehicle)).astype(np.float64)
    # Define the labels vector1: car 0: no car
    y = np.hstack((np.ones(len(features_vehicle)), np.zeros(len(features_non_vehicle))))
    #print(y.shape)

    # split test data
    rnd = np.random.randint(0, 100)
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
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    if bPersistant:
        save_object(svc,data_file)
        print("Saved classifier data")

    return svc

def get_car_windows(classifier):


    pass



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
    #load training data
    vehicles = load_images_rec('training/vehicles',maxCount=DATA_LIMIT)
    print("Vehicles: "+str(len(vehicles)))
    non_vehicles = load_images_rec('training/non-vehicles',maxCount=DATA_LIMIT)
    print("Non-Vehicles: " + str(len(non_vehicles)))

    #train
    classifier = train_classifier(vehicles,non_vehicles,C_PERSISTANT_CLASSIFIER_DATA)


    #clear memory
    vehicles = None
    non_vehicles = None
    gc.collect()

    ## Preparation Done ---- Looping images / video from here on
    if Video:
        setResetAllways(False)
        project_video_output = "project_video_output.mp4"
        project_video_input = VideoFileClip("project_video.mp4")

        #define partial function for arguments
        video_frame = partial(process_frame)

        processed_project_video = project_video_input.fl_image(video_frame)
        processed_project_video.write_videofile(project_video_output, audio=False)

    else:
        setResetAllways(True) # we need to reset means and medians as the images are independent
        images = load_images('test_images/')
        for im in images:
            res = process_frame(im)

    return 0


if __name__ == '__main__':
    main()