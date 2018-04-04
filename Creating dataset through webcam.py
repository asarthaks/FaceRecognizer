import cv2
import matplotlib.pyplot as plt
import os


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

#initializing the camera
webcam = cv2.VideoCapture(0)
print(webcam.isOpened())

#initializing the detector
detector = FaceDetector("haarcascade_frontalface_default.xml")


#detecting faces, normalizing and saving
folder = "people/"+ input('Person: ').lower() #input name
cv2.namedWindow("Face Recog", cv2.WINDOW_AUTOSIZE)
if not os.path.exists(folder):
    os.mkdir(folder)
    counter = 0
    timer = 0
    while counter < 50 :
        _, frame = webcam.read()
        faces_coord = detector.detect(frame) #detect
        if len(faces_coord) and timer % 700 == 50: #every second or so
            faces = normalize_faces(frame, faces_coord) #norm pipeline
            cv2.imwrite(folder + '/' + str(counter) + '.jpg', faces[0])   #saved face in notebook
            cv2.imshow('Images Saved:' + str(counter),faces[0])
            counter += 1

        draw_rectangle(frame, faces_coord) #rectangle around face
        cv2.imshow("Face Recog", frame) #live feed in external
        cv2.waitKey(50)
        timer += 50
        cv2.destroyAllWindows()

else:
    print("this name already exists.")


#camera release   
webcam.release()
