import numpy
import cv2

cap = cv2.VideoCapture(0)
#Change path as per folder structure
faceCascade = cv2.CascadeClassifier("C:\Work\Rohit\PythonCV\haarcascade_frontalface_default.xml")

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #BGR to Gray    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    )

    #Draw rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #Display image with box and print no of faces in terminal
    cv2.imshow("Faces found", frame)
    print("Found {0} faces!".format(len(faces)))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
