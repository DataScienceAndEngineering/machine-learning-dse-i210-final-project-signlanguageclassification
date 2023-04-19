#import libraries
import cv2 as cv

#function to get video 
def get_video():
    #define video capture from camera feed 
    cap = cv.VideoCapture(0)
    #check if video is unable to be obtained, and print message 
    if cap.isOpened() == False:
        print('Error opening video stream')
    #while video is being captured
    while (cap.isOpened()):
        #read the video from camera feed
        ret, frame = cap.read()
        #if video is being returned 
        if ret == True:
            #launch window and display 
            cv.imshow('Sign Language Interpreter',frame)
            #quit if q is pressed 
            if cv.waitKey(25) & 0xFF == ord('q'):
                break 
        #quit if video is not returned 
        else:
            break 
    #release video capture and destroy imshow windows 
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    get_video()