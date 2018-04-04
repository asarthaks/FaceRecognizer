import cv2
import cv2.face
import numpy as np
import matplotlib.pyplot as plt
import os


def collect_dataset():
    images = []
    labels = []
    labels_dic = {}
    people = [person for person in os.listdir('people/')]
    for i,person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir('people/' +person):
            images.append(cv2.imread('people/' + person + '/' + image, 0))
            labels.append(i)
    return (images,np.array(labels), labels_dic)



images, labels, labels_dic = collect_dataset()

rec_eig = cv2.face.EigenFaceRecognizer_create()     
rec_eig.train(images, labels)

#need at least two people
rec_fisher = cv2.face.FisherFaceRecognizer_create()
rec_fisher.train(images, labels)

rec_lbph = cv2.face.LBPHFaceRecognizer_create()
rec_lbph.train(images, labels)

print("Models Trained Succesfully")


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


def cut_faces(image, faces_coord):
    faces = []
    for (x,y,w,h) in faces_coord:
        w_rm = int(0.2 * w/2)
        
        faces.append( image[y: y+h, x+w_rm: x+w - w_rm])
        
    return faces


def normalize_intensity(images):
    images_norm = []
    for image in images:
        is_color = len(image.shape)==3
        if is_color:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm

def resize(images, size=(100,100)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, interpolation = cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, interpolation = cv2.INTER_CUBIC)

        images_norm.append(image_norm)

    return images_norm


def normalize_faces(frame, faces_coord):
    faces = cut_faces(frame, faces_coord)
    faces = normalize_intensity(faces)
    faces = resize(faces)
    return faces

def draw_rectangle(image, coords):
    for(x, y, w, h) in coords:
        w_rm = int(0.2 *w/2)
        cv2.rectangle(image, (x+w_rm, y), (x + w - w_rm, y + h), (150, 150, 0), 8)



#initializing camera and detector
detector = FaceDetector("haarcascade_frontalface_default.xml")
webcam = cv2.VideoCapture(0)
print(webcam.isOpened())



#live
cv2.namedWindow('Live Detection', cv2.WINDOW_AUTOSIZE)
while True:
    _, frame = webcam.read()
    frame = cv2.flip(frame,1)
    faces_coord = detector.detect(frame, True) #detect more than one face
    if len(faces_coord):
        faces = normalize_faces(frame, faces_coord) #norm pipeline
        for i, face in enumerate(faces): # for each detected face
            collector = cv2.face.StandardCollector_create()
            rec_lbph.predict_collect(face, collector)
            conf = collector.getMinDist()
            pred = collector.getMinLabel()
            threshold = 140
            if conf < 100:
                print("prediction: " + labels_dic[pred].capitalize() + "\nConfidence: " + str(conf))
                cv2.putText(frame, labels_dic[pred].capitalize(),
                            (faces_coord[i][0], faces_coord[i][1] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 3, (66,53,243), 2)
            else:
                 print("prediction: unknown" )  
                 cv2.putText(frame, 'Unknown',
                            (faces_coord[i][0], faces_coord[i][1] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 3, (66,53,243), 2)
            draw_rectangle(frame, faces_coord) #rectangle around face
        cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, (66,53,243), 2, cv2.LINE_AA)
    cv2.imshow("live detection", frame)
    if cv2.waitKey(40) & 0xFF ==27:
        cv2.destroyAllWindows()
        break



#release webcam
webcam.release()




