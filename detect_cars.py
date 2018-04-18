# imports
from helper_functions import *
# ---------------------------------

C_DEBUG = False
VID_DEBUG = False

_EXPORT_PICTURES_ = True


def process_frame(im):



    if C_DEBUG:
        show_image(im)


#    if C_DEBUG:
#        plt.plot(hist)
#        show_image(fig2data(plt.gcf()))
#        plt.clf()

    processed_img = im

    return processed_img


def main():
    # main, do something
    global C_DEBUG # may be changed here

    Video = True

    if _EXPORT_PICTURES_:
        setWriteAllImages(True)
        C_DEBUG = True
        Video = False

    #C_DEBUG = True
    images = load_images('test_images/')

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
        for im in images:
            res = process_frame(im)
            show_image(res)

    return 0


if __name__ == '__main__':
    main()