import cv2
import matplotlib.pyplot as plt

#defining the classes

class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)

    def detect(self, image, biggest_only = True) :
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30,30)
        biggest_only = True
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
                cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
                cv2.CASCADE_SCALE_IMAGE

        faces_coord = self.classifier.detectMultiScale(frame, scaleFactor= scale_factor, minNeighbors = min_neighbors,
                                        minSize = min_size, flags = flags)

        return faces_coord


#initializing the camera
webcam = cv2.VideoCapture(0)
print(webcam.isOpened())

#initializing the detector
detector = FaceDetector("haarcascade_frontalface_default.xml")



#drawing rectangle and displaying frame
while webcam.isOpened():
        _, frame = webcam.read()
        frame = cv2.flip(frame,1)     #inverting the image
        faces_coord = detector.detect(frame)
        if len(faces_coord):
            for (x,y,w,h) in faces_coord:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (150,150,0),8)
            
        cv2.imshow('Face Detector',frame)
        #clear_output(wait = True)
        
        #code 27 is ESC key
        if cv2.waitKey(20) & 0xFF ==27:
            break



#camera release   
webcam.release()
