import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from gpiozero import LED

unlock = LED(17)
lock = LED(18)

#loads model
model = tf.keras.models.load_model('mask_detection_model_v7')

# this enables facial detection for OpenCV
face_detection = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# open the default camera
camera = cv.VideoCapture(1)

#camera.set(3, 640)
#camera.set(4, 480)

if not camera.isOpened():
    print("Cannot open camera")
    exit()

if not camera:
    print ("!!! Failed VideoCapture: unable to open device 0")
    sys.exit(1)

# 0 is mask, 1 is no mask
labels_dict={0:'MASK',1:'NO MASK'}
# green box, red box
color_dict={0:(0,255,0),1:(0,0,255)}

while True:
    print("in while loop")

    # camera.read() returns 2 variables, one matrix of the image one video
    matrix,frame = camera.read()
    print("reading cam")

    # the video in tensor or matrix form
    #print(matrix)
    # the actual video stream 
    #print(frame)

    # convert frame into grey scale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # get faces from gray video frame
    faces = face_detection.detectMultiScale(gray,1.3,5)  

    if (len(faces) == 0):
        unlock.off()
        lock.on()

    for (x,y,w,h) in faces:
        # crops the region which the face is found within the gray frame
        face_img = gray[y:y+w,x:x+w]
        # resizing image to what the nn was trained for
        # cv.resize(face_img,(50,50) returns a tensor of the image
        resized = cv.resize(face_img,(100,100))
        
        # dividing image by 255 because each pixel only has a max value of 255
        # so by doing this each value returned in the resize tensor will now be
        # between 0-1 making it easier for our nn to manage
        normalized = resized/255.0
        # reshapes size of array to 4D since convnets takes in 4d array 
        reshaped = np.reshape(normalized,(1,100,100,1))
        # images shown in tensors
        # print(reshaped)
        # how many array dimensions
        # print(reshaped.ndim)

        # shows the image that is sent to the nn
        # plt.imshow(resized,cmap="gray")
        # plt.show()

        # runs image of face that was caputures through the model
        result = model.predict(reshaped)
        #print(result)
        # returns index of mask status
        label = np.argmax(result, axis=1)[0]
        print(labels_dict[label]," ",result[0][0]*100)

        # if(int(result[0][0]*100) < 97):
        #     label = 1
        if(label == 0):
             unlock.on()
             lock.off()
        else:
             unlock.off()
             lock.on() 

        cv.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],2)
        cv.rectangle(frame,(x,y-40),(x+w,y),color_dict[label],-1)
        cv.putText(frame, labels_dict[label], (x, y-10),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv.putText(frame, str(round(result[0][0]*100, 2)), (x+125, y-10),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv.imshow('Capturing', frame)

    key = cv.waitKey(1)

    if key == ord('q'):
        break

camera.release()
cv.destroyAllWindows()


