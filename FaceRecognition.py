import numpy as np
import os
import cv2
from scipy.misc import face

# C:\Users\Pep\Pictures\Camera Roll\Privat

#cap = cv2.VideoCapture(0)

subjects = ["", "Josep", "Ferran"]


def detect_face(img):
    width_d, height_d = 250, 250 

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if (len(faces) == 0):
        return None, None

    imageNp = np.array(img, 'uint8')

    (x, y, w, h) = faces[0]
    for (x, y, w, h) in faces:
        faces = cv2.resize(imageNp[y:y + h, x:x + w], (width_d, height_d))
    return gray[y:y + w, x:x + h], faces[0]

def prepare_training_data():

    dirs = os.listdir("C:\Users\Pep\Pictures\Camera Roll")
    faces = []
    labels = []
    for dir_name in dirs:
        if not dir_name.startswith("P"):
            continue;
        label = int(dir_name.replace("P", ""))
        subject_dir_path = "C:\Users\Pep\Pictures\Camera Roll" + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue;
            image_path = subject_dir_path + "/" + image_name
            print(image_path)
            image = cv2.imread(image_path)
            face, rect = detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels

print("Preparing data...")
faces, labels = prepare_training_data()
print("Data prepared")
c = 0

for i in faces:
    try:

        height, width = i.shape[:2]
        print(height)
        print(width)
    except:
        faces.pop(c)
        labels.pop(c)
        print("An invalid Image has been detected...continuing")
    c += 1
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

model = cv2.face.EigenFaceRecognizer_create()
model.train(faces, np.array(labels))
#------------------------------------------
""" 
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
"""
