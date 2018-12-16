import cv2
import matplotlib.pyplot as plt


vidcap = cv2.VideoCapture('vid1.mp4')
success,image = vidcap.read()
count = 0
success = True
images = []
while success:
    try :
        vidcap.set(1, count)
        success,image = vidcap.read()

        #cv2.imwrite("frame%d.jpeg" % count, image)     # save frame as JPEG file
        #img_before = cv2.imread("frame%d.jpeg" % count)
        #img_before = image.astype(np.uint8)


        test1 = image

        #image_array[count] = imread("/home/me/Pictures/img1.png", 1);




        def detect_faces(f_cascade, colored_img, scaleFactor=1.1):
            # just making a copy of image passed, so that passed image is not changed
            img_copy = colored_img.copy()

            # convert the test image to gray image as opencv face detector expects gray images
            gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

            # let's detect multiscale (some images may be closer to camera than others) images
            faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)

            # go over list of faces and draw them as rectangles on original colored img
            for (x, y, w, h) in faces:

                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

            return img_copy

        def convertToRGB(img):
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #load test iamge
        result_image = test1.copy()
        #convert the test image to gray image as opencv face detector expects gray images
        gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)

        #plt.imshow(gray_img, cmap='gray')


        #test2 = cv2.imread('data/test1.jpg')
        haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        # call our function to detect faces
        faces_detected_img = detect_faces(haar_face_cascade, test1)

        # convert image to RGB and show image
        #plt.imshow(convertToRGB(faces_detected_img))


        #let's detect multiscale (some images may be closer to camera than others) images
        faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

        #print the number of faces found
        print('Faces found: ', len(faces))

        age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

        for (x, y, w, h) in faces:
            #cv2.rectangle(test1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            sub_face = test1[y:y + h, x:x + w]
            sub_face = cv2.GaussianBlur(sub_face, (23, 23), 30)
            result_image[y:y + sub_face.shape[0], x:x + sub_face.shape[1]] = sub_face
            face_img = sub_face[y:y + h, h:h + w].copy()


            cv2.rectangle(test1, (x, y), (x + w, y + h), (255, 255, 0), 2)
            # Get Face
            face_img = test1[y:y + h, h:h + w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            """
            #Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            print("Age Range: " + age)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            overlay_text = str(age)
            
            cv2.putText(result_image, overlay_text, (x, y), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            plt.imshow(convertToRGB(result_image))
            """


        images.append(result_image)
        count += 1
        print('Read a new frame: ', success)
    except :
        success = False


frame  = images[0]
height, width, layers = frame.shape
video = cv2.VideoWriter("sd21.avi", -1, 1, (width, height))
for image in images :
    video.write(image)
cv2.destroyAllWindows()
video.release()



