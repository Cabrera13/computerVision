import numpy as np
import os
import cv2
from scipy.misc import face

# C:\Users\Pep\Pictures\Camera Roll\Privat

#cap = cv2.VideoCapture(0)

subjects = ["", "Josep", "Ferran"]

#function to detect face using OpenCV
def detect_face(img):
    width_d, height_d = 250, 250  # Declare your own width and height

    #convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow: Haar classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    #let's detect multiscale images(some images may be closer to camera than others)
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    #under the assumption that there will be only one face,
    #extract the face area
    imageNp = np.array(img, 'uint8')

    #gray = cv2.resize(gray, (250,100))

    (x, y, w, h) = faces[0]
    for (x, y, w, h) in faces:
        ########################################
        # The line to be changed by cv2.resize()
        ########################################
        faces = cv2.resize(imageNp[y:y + h, x:x + w], (width_d, height_d))

    return gray[y:y + w, x:x + h], faces[0]

    #return only the face part of the image


# this function will read all persons' training images, detect face from each image
# and will return two lists of exactly same size, one list
# of faces and another list of labels for each face

def prepare_training_data():


    # ------STEP-1--------
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir("C:\Users\Pep\Pictures\Camera Roll")

    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # let's go through each directory and read images within it
    for dir_name in dirs:

        # our subject directories start with letter 's' so
        # ignore any non-relevant directories if any
        if not dir_name.startswith("P"):
            continue;

        # ------STEP-2--------
        # extract label number of subject from dir_name
        # format of dir name = slabel
        # , so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("P", ""))

        # build path of directory containing images for current subject subject
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = "C:\Users\Pep\Pictures\Camera Roll" + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        # ------STEP-3--------
        # go through each image name, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name
            print(image_path)
            # read image
            image = cv2.imread(image_path)
            # display an image window to show the image
            #cv2.imshow("Training on image...", image)
            # detect face
            face, rect = detect_face(image)

            # ------STEP-4--------
            # for the purpose of this tutorial
            # we will ignore faces that are not detected
            if face is not None:
                # add face to list of faces
                faces.append(face)
                #faces.append(cv2.resize(np.array[y:y + h, x:x + w], (width_d, height_d))


                # add label for this face
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data()
print("Data prepared")

# print total faces and
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
#cv2.face.createEigenFaceRecognizer()
#face_recognizer = cv2.face.createEigenFaceRecognizer()
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