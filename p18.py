import cv2,time
import numpy as np
from os import listdir #list directory is used when we have to fetch the data from a directory
from os.path import isfile,join
data_path = "C:\\imp files\\JUNK\\"
# for loop used coz if there is a file in the folder then it will join with the variable f
onlyfiles = [f for f in listdir(data_path)if isfile(join(data_path,f))]
# now we define two variables as list
training_data,labels = [],[]
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i] #only files are arrenged according to the integer value i
    images = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE) #since we are working with the gray images so we have to tell the system about that
    training_data.append(np.asarray(images,dtype=np.uint8)) #in the list we have appended images with unsigned integer of 8 bit data type as array
    labels.append(i)
labels = np.asarray(labels,dtype=np.int32) #just defining integer of 32 bit is used as data type
model = cv2.face.LBPHFaceRecognizer_create() #just defined linear binary phase histogram face recognizer it basically train the data set
model.train(np.asarray(training_data),np.asarray(labels)) #training the dataset so that it can recognize the correct face
print("DATA TRAINED ! ! !")
face_classifier = cv2.CascadeClassifier("C:\\imp files\\haarcascade_frontalface_default.xml")
def face_detector(img,size = 0.5):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is(): #If it has no image then it will return the blank image
        return img,[]
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        o = img[y:y+h,x:x+w] #Region of interest in image which is going to used
        o = cv2.resize(o,(200,200))
    return img,o
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read() #Read from camera
    image,face = face_detector(frame)
    try: # It allows us to test a block of code with certain amount of error
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result = model.predict(face) # It predict the face is correct or not
        if result[1] < 400:
            belief = int(100*(1-(result[1])/300)) # Formula for percentage of matching the face
            display_string = str(belief)+ "% Confirmation"
        cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_DUPLEX,1,(0,255,150),2)
        if belief > 80:
            cv2.putText(image,"Its Ankur", (390, 480), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0),2)
            cv2.imshow("Image Viewer",image)
        else:
            cv2.putText(image, "Intruder", (210, 380), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0 , 255), 2)
            cv2.imshow("Image Viewer", image)
    except: # It is used to block and allow us to handle the error
        cv2.putText(image, "FACE MISSING", (240, 420), cv2.FONT_HERSHEY_DUPLEX, 1, (180, 0 , 255), 2)
        cv2.imshow("Image Viewer", image)
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()