#import libraries
import cv2 as cv
import mediapipe as mp

#function for detecting the hand 
def findHand(img):
    #define mediapipe hands model
    mpHands = mp.solutions.hands
    #hands object to detect and track hands 
    hands = mpHands.Hands()
    #convert image to grayscale as preprocessing step
    g_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #return detected hands 
    return hands.process(g_img)

#function for finding the rectangular bound of the hand detected 
def find_rectangle(img, results):
    #use the height and width of the img as a starting point for the x_min, y_min 
    h,w,c = img.shape
    #constant to increase the size of the box 
    constant = 100
    #check if hand is detected via landmarks 
    if results.multi_hand_landmarks:
        #loop through the detected hand landmarks (landmarks include x,y,z positions)
        for handLMs in results.multi_hand_landmarks:
            #define min and max values for coordinates of rectangle
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            #loop through the x,y,z positions of the landmarks 
            for lm in handLMs.landmark:
                #logic for finding the upper left corner and lower right corner
                #based on the outer points of the hand 
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
        return x_min - constant, y_min - constant, x_max + constant, y_max + constant
    else:
        return None, None, None, None

#function for getting a square image as input to the sign language interpreter model  
def get_cropped_image(img,x1,y1,x2,y2):
    #constant to add to build a square
    constant = 200
    #find the center of the rectangle
    x, y = int((x2+x1)/2),int((y2+y1)/2)
    #try to slice the numpy array 
    try:
        cropped_image = img[y-constant:y+constant,x-constant:x+constant]
    #if IndexError since hand is far off the center of the camera view, set cropped image to none
    except IndexError:
        cropped_image = None
    #return the cropped image 
    return cropped_image



#main function call 
def main():
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
            #detect hands 
            detection_results = findHand(frame)
            #find rectangular bound around detected hand
            x1,y1,x2,y2 = find_rectangle(frame,detection_results)

            #if bound is found 
            if x1:
                #get cropped image for model input if possible 
                cropped_img = get_cropped_image(frame,x1,y1,x2,y2)

                #TEMPORARY IMSHOW FOR PROOF OF CONCEPT AND DEBUGGING#
                if cropped_img is not None:
                    cv.imshow('Cropped Image',cropped_img)

                #draw rectangular bound in frame if found 
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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

#entry 
if __name__ == '__main__':
    main()