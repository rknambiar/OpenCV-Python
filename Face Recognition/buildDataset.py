import sys
import os
import cv2


def create_dataset():
    folder = "people/" + input('Person: ').lower()
    cv2.namedWindow("Python Face Reco Tutorial",cv2.WINDOW_AUTOSIZE)
    cap = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier("C:\Work\Rohit\PythonCV\haarcascade_frontalface_default.xml")

    if not os.path.exists(folder):
        os.mkdir(folder)
        counter  = 0
        timer = 0
        while counter < 10:
            ret, frame = cap.read()
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

            if len(faces) and timer % 700 == 50:
                faces_crop = cut_faces(frame,faces)
                faces_eqal = norm_intensity(faces_crop)
                faces_res  = resize(faces_eqal)
                cv2.imwrite(folder + '/' + str(counter) + '.jpg',faces_res[0])
                cv2.imshow("Faces saving",faces_res[0])
                print("Image saved:" + str(counter))
                counter += 1
            draw_rectangle(frame,faces)
            cv2.imshow("Live output",frame)
            cv2.waitKey(50)
            timer += 50
        cv2.destroyAllWindows()
    else:
        print ("This name already exists.")
        

def draw_rectangle(frame,faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


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
    print("Hello")
    create_dataset()

