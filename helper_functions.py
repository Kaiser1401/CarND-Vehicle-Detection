# helper functions for lane finding.
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import fnmatch
import pickle

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


_RESET_ALLWAYS_ = False
_WRITE_ALL_IMAGES_ = False
_WRITE_IMAGE_BASE_NAME_ = "output_images/i_"
_IMG_FILE_COUNTER_ = 0
_NO_SHOW_ = False

def save_object(obj,filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2): # as from the first project
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):   # as from the first project
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def gray(img_in,conversion=cv2.COLOR_RGB2GRAY): # use COLOR_BGR2GRAY if loaded by cv2
    return cv2.cvtColor(img_in, conversion)


def write_image(image, path):
    if len(image.shape) == 3:
        if image.shape[2] == 4: #rgba
            cv2.imwrite(path, image)
            return
        else:
            cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return
    cv2.imwrite(path, image)

def load_images(folder):
    fileList = os.listdir(folder)
    imgList = []
    for f in fileList:
        imgList.append(load_image(folder+f))
    return imgList

def recursive_glob(rootdir='.', pattern='*'): # from https://gist.github.com/whophil/2a999bcaf0ebfbd6e5c0d213fb38f489
    """Search recursively for files matching a specified pattern.
    Adapted from http://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python
    """
    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches

def load_images_rec(folder,ext=".png",maxCount=0):
    files = recursive_glob(folder,"*"+ext)
    imgList = []
    c = 0
    for f in files:
        if maxCount > 0:
            if c >= maxCount:
                return imgList
        imgList.append(load_image(f))
        c+=1
    return imgList

def fig2data(fig): # from http://www.icare.univ-lille1.fr/wiki/index.php/How_to_convert_a_matplotlib_figure_to_a_numpy_array_or_a_PIL_image
    """
      @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
      @param fig a matplotlib figure
      @return a numpy 3D array of RGBA values
      """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    #buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)

    # to bgr
    #buf = np.dstack((buf[:,:,3], buf[:,:,2], buf[:,:,1]))

    return buf

def load_image(path):
    try:
        return mpimg.imread(path)
    except Exception as e:
        print ("EX: "+str(e)+" p:"+path)


def binFilter(img,thres): # multi channel binary-AND-Filter
    fullValue = 255
    try:
        binary = np.ones_like(img[:,:,1]) # shape!!
    except:
        binary = np.ones_like(img)  # shape!!

    binary *= fullValue
    #binary[0,0]=0 # one black pixel for visualisation

    #show_image(binary,True)

    for i in range(len(thres)):
        if len(thres) > 1:
            tmp = img[:,:,i]
        else:
            tmp = img
        #show_image(tmp, True)
        thrs = thres[i]
        binary[(tmp < thrs[0]) | (tmp > thrs[1]) | (binary < 1)] = 0
        #show_image(binary, True)

    return binary

def show_image(image,bBW=False):

    if _WRITE_ALL_IMAGES_:
        write_image(image, _WRITE_IMAGE_BASE_NAME_ + str(_IMG_FILE_COUNTER_) + ".png")
        global _IMG_FILE_COUNTER_
        _IMG_FILE_COUNTER_ += 1
        return

    if _NO_SHOW_:
        return

    if bBW:
        plt.imshow(image,cmap='gray')
    else:
        plt.imshow(image)

    plt.show(block=True)

_DO_MEDIAN_ = True

_median_vars = {} #place to hold old values for medians
_median_idx  = {} #place to hold old values for medians

def setResetAllways(value):
    global _RESET_ALLWAYS_
    _RESET_ALLWAYS_= value

def setNoShow(value):
    global _NO_SHOW_
    _NO_SHOW_= value

def getResetAllways():
    return _RESET_ALLWAYS_

def setWriteAllImages(value):
    global _WRITE_ALL_IMAGES_
    _WRITE_ALL_IMAGES_ = value

def runningMedian(currentMeassure, countMeasurements, name):
    if (not name in _median_vars) or _RESET_ALLWAYS_:
        _median_vars[name] =  np.repeat(np.expand_dims(currentMeassure, axis=0), countMeasurements, axis=0)
        _median_idx[name] = 0

    # insert current data
    _median_vars[name][_median_idx[name]]=currentMeassure

    # find next pos
    _median_idx[name]+=1
    if _median_idx[name] >= countMeasurements:
        _median_idx[name] = 0

    if _DO_MEDIAN_:
        # find median
        return np.median(_median_vars[name],axis=0)
    else:
        # find mean
        return np.mean(_median_vars[name],axis=0)


# from/based on Lesson 0 #35
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm= 'L2-Hys',
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm= 'L2-Hys',
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features

# from/based on Lesson 0 #35
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# from/based on Lesson 0 #35
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

# from/based on Lesson 0 #22
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256), hog_orient = 6,
                     hog_pix_per_cell = 8,hog_cell_per_block = 2):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for image in imgs:
        #print(image.shape)

        # Read in each one by one
        #image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        im_features=[]
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else: feature_image = np.copy(image)
        # Apply bin_spatial() to get spatial color features
        im_features.append(bin_spatial(feature_image, size=spatial_size))
        # Apply color_hist() also with a color space option now
        im_features.append(color_hist(feature_image, nbins=hist_bins))

        hog_features = []
        for i in range(feature_image.shape[2]): #for each channel
            hog_features.append(get_hog_features(feature_image[:,:,i],hog_orient,hog_pix_per_cell,hog_cell_per_block,False,True))

        hog_features = np.ravel(hog_features)
        im_features.append(hog_features)

        # Append the new feature vector to the features list
        features.append(np.concatenate(im_features))
    # Return list of feature vectors
    return features

# from/based on Lesson 0 #32
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# from/based on Lesson 0 #34
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   orient=9,
                   pix_per_cell=8, cell_per_block=2 ):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()

        # make a list with one entry to reuse extract features function
        single_im_list=[]
        single_im_list.append(test_img)
        features = extract_features(single_im_list, cspace=color_space,
                         spatial_size=spatial_size, hist_bins=hist_bins,
                         hog_orient=orient, hog_pix_per_cell=pix_per_cell,
                         hog_cell_per_block=cell_per_block)


        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows

# from/based on Lesson 0 #35
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

# from/based on Lesson 0 #35
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    boundingbox = []
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]

    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YUV')

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                # cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                boundingbox.append(
                    ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
    return boundingbox


# from/based on Lesson 0 #32
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# from/based on Lesson 0 #37
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))

        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes

# from/based on Lesson 0 #37
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

# from/based on Lesson 0 #37
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img

#images_vehicle[0],hog_orient,hog_pix_per_cell,hog_cell_per_block,vis=True
def get_hog_img_3_chan(img,orient,pix,cell):
    ch1 = img[:, :, 0]
    ch2 = img[:, :, 1]
    ch3 = img[:, :, 2]
    tmp, hog1 = get_hog_features(ch1, orient, pix, cell, vis=True)
    tmp, hog2 = get_hog_features(ch2, orient, pix, cell, vis=True)
    tmp, hog3 = get_hog_features(ch3, orient, pix, cell, vis=True)
    #print (hog1.shape)
    hog_img = np.vstack((hog1, hog2, hog3))
    max=np.max(hog_img)
    hog_img=hog_img*(1.0/max).astype(np.float32)
    print (hog_img.shape)
    return hog_img