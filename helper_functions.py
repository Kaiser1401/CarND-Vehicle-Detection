# helper functions for lane finding.
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from moviepy.editor import VideoFileClip
from functools import partial


_RESET_ALLWAYS_ = False
_WRITE_ALL_IMAGES_ = False
_WRITE_IMAGE_BASE_NAME_ = "output_images/i_"
_IMG_FILE_COUNTER_ = 0

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

def load_images_recursive(folder):
    # do stuff -> https://stackoverflow.com/questions/44589915/read-images-from-multiple-folders-in-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    pass


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
    return mpimg.imread(path)

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



