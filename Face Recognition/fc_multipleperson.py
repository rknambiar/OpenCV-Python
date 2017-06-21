import os
import sys
import cv2
import numpy as np


def collect_dataset():
    images = []
    labels = []
    labels_dic = {}
    people = [person for person in os.listdir("people/")]
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("people/" + person):
            images.append(cv2.imread("people/" + person + '/' + image,0))
            labels.append(i)
    return(images,np.array(labels), labels_dic)


def train_dataset():
    images, labels, labels_dic = collect_dataset()

    rec_eig = cv2.face.createEigenFaceRecognizer()
    rec_eig.train(images,labels)

    print("Models Trained Succesfully")
    return(rec_eig,labels_dic)

def face_recognition(recognizer,labels_dic):
    cap = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier("C:\Work\Rohit\PythonCV\haarcascade_frontalface_default.xml")

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        #BGR to Gray    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        biggest_only = True     
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_FIND_BIGGEST_OBJECT | \
                  cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
                  cv2.CASCADE_SCALE_IMAGE
        
        )
        if len(faces):
            #Iterate through faces                        
            faces_crop = cut_faces(frame,faces)
            faces_eqal = norm_intensity(faces_crop)
            faces_res  = resize(faces_eqal)
            for i,face in enumerate(faces_res):
                name = predict(face,recognizer,labels_dic)
                (x, y, w, h)  = faces[i]
                #Draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)

                
        #Display image with box and print no of faces in terminal
        cv2.imshow("Faces live", frame)
            
                        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def predict(face,recognizer,labels_dic):
       
    if cv2.__version__ >= "3.1.0":
        collector = cv2.face.StandardCollector_create()
        recognizer.predict_collect(face,collector)
        conf = collector.getMinDist()
        pred = collector.getMinLabel()
    else:
        collector = cv2.face.MinDistancePredictCollector()
        recognizer.predict(face,collector)
        conf = collector.getDist()
        pred = collector.getLabel()
    print ("Eigen Faces-> Prediction: " + labels_dic[pred].capitalize() + "   Confidence: " + str(round(conf)))
    cv2.imshow("Face Cropped",face)
    return (labels_dic[pred])


def cut_faces(image,faces_coord):
    faces = []

    for (x, y, w, h) in faces_coord:
        w_rm = int(0.2 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])

    return faces

def norm_intensity(images):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)           
        images_norm.append(cv2.equalizeHist(image))
    return images_norm

def resize(images, size=(100, 100)):
    images_resize = []
    for image in images:
        if image.shape < size:
            image_resize = cv2.resize(image,size,interpolation = cv2.INTER_AREA)
        else:
            image_resize = cv2.resize(image,size,interpolation = cv2.INTER_CUBIC)
        images_resize.append(image_resize)
    return images_resize

if __name__== "__main__":   
    recognizer,labels_dic = train_dataset()    
    face_recognition(recognizer,labels_dic)
