import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    test1 = frame


    def detect_faces(f_cascade, colored_img, scaleFactor=1.1):
        img_copy = colored_img.copy()

        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

        faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return img_copy


    def convertToRGB(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    # load test iamge
    result_image = test1.copy()
    gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)

    # plt.imshow(gray_img, cmap='gray')

    # test2 = cv2.imread('data/test1.jpg')
    haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces_detected_img = detect_faces(haar_face_cascade, test1)

    # plt.imshow(convertToRGB(faces_detected_img))

    faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    print('Faces found: ', len(faces))

    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

    for (x, y, w, h) in faces:
        # cv2.rectangle(test1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        sub_face = test1[y:y + h, x:x + w]
        ellipse = cv2.fitEllipse(sub_face)
        sub_face = cv2.GaussianBlur(ellipse, (23, 23), 30)
        result_image[y:y + sub_face.shape[0], x:x + sub_face.shape[1]] = sub_face
        face_img = sub_face[y:y + h, h:h + w].copy()

        cv2.rectangle(test1, (x, y), (x + w, y + h), (255, 255, 0), 2)
        # Get Face
        face_img = test1[y:y + h, h:h + w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)





    cv2.imshow('frame', result_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
