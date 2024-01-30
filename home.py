import cv2
import numpy as np
import dlib
import uuid
from math import hypot

def empty(a):
    pass

def createBox(img, points, scale=5, masked=False, cropped=True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [points], (255, 255, 255))
        img = cv2.bitwise_and(img, mask)

    if cropped:
        bbox = cv2.boundingRect(points)
        x, y, w, h = bbox
        imgCrop = img[y:y+h, x:x+w]
        imgCrop = cv2.resize(imgCrop, (0, 0), None, scale, scale)
        return imgCrop
    else:
        return mask

print("Choose an option:")
print("1. Lips Color")
print("2. Pig Nose")
print("3. Devil Horns")
print("4. Mustache")
print("0. Exit")

choice = input("Enter your choice (1, 2, 3, 4 or 0): ")

if choice == '1':
    webcam = True
    cap = cv2.VideoCapture(0)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    cv2.namedWindow("BGR")
    cv2.resizeWindow("BGR", 640, 240)
    cv2.createTrackbar("Blue", "BGR", 0, 255, empty)
    cv2.createTrackbar("Green", "BGR", 0, 255, empty)
    cv2.createTrackbar("Red", "BGR", 0, 255, empty)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("failed")
            break

        imOriginal = frame.copy()
        imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(imGray)

        for face in faces:
            x1, y1 = face.left(), face.top() #Retrieves the coordinates of the top-left (x1, y1) and bottom-right
            x2, y2 = face.right(), face.bottom() #bottom-right (x2, y2) corners of the bounding box around the detected face
            landmarks = predictor(imGray, face)
            myPoints = []
            for n in range(68):

                x = landmarks.part(n).x
                y = landmarks.part(n).y
                myPoints.append([x, y])


            myPoints = np.array(myPoints)
            imgLips = createBox(frame, myPoints[48:61], 3, masked=True, cropped=False)



            imgColorLips = np.zeros_like(imgLips)
            b = cv2.getTrackbarPos('Blue', 'BGR')
            g = cv2.getTrackbarPos('Green', 'BGR')
            r = cv2.getTrackbarPos('Red', 'BGR')


            imgColorLips[:] = b, g, r
            imgColorLips = cv2.bitwise_and(imgLips, imgColorLips)

            imgColorLips = cv2.GaussianBlur(imgColorLips, (7, 7), 10)
            imgColorLips = cv2.addWeighted(imOriginal, 1, imgColorLips, 0.4, 0)

            cv2.imshow('BGR', imgColorLips)
            cv2.imshow('Lips', imgLips)


        k = cv2.waitKey(1)


        if k % 256 == 27:
            print("escape")
            break
        elif k % 256 == 32:
            img_name = "screenshot_{}.png".format(str(uuid.uuid4())[:8])
            cv2.imwrite(img_name, imgColorLips)  # Save the screenshot from "BGR" window
            print("screenshot taken")
            cv2.waitKey(100)

    cap.release()
    cv2.destroyAllWindows()

elif choice == '2':
    # Pig Nose functionality (existing code)
    cap = cv2.VideoCapture(0)
    nose_image = cv2.imread("pig_nose.png")
    _, frame = cap.read()
    rows, cols, _ = frame.shape
    nose_mask = np.zeros((rows, cols), np.uint8)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    cv2.namedWindow("ss app")
    img_counter = 0

    while True:
        _, frame = cap.read()
        nose_mask.fill(0)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(frame)
        for face in faces:
            landmarks = predictor(gray_frame, face)

            # Nose coordinates
            top_nose = (landmarks.part(29).x, landmarks.part(29).y)
            center_nose = (landmarks.part(30).x, landmarks.part(30).y)
            left_nose = (landmarks.part(31).x, landmarks.part(31).y)
            right_nose = (landmarks.part(35).x, landmarks.part(35).y)

            nose_width = int(hypot(left_nose[0] - right_nose[0],
                                   left_nose[1] - right_nose[1]) * 1.7)
            nose_height = int(nose_width * 0.77)

            # New nose position
            top_left = (int(center_nose[0] - nose_width / 2),
                        int(center_nose[1] - nose_height / 2))
            bottom_right = (int(center_nose[0] + nose_width / 2),
                            int(center_nose[1] + nose_height / 2))

            # Adding the new nose
            nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
            nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
            _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)

            nose_area = frame[top_left[1]: top_left[1] + nose_height,
                        top_left[0]: top_left[0] + nose_width]
            nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
            final_nose = cv2.add(nose_area_no_nose, nose_pig)

            frame[top_left[1]: top_left[1] + nose_height,
            top_left[0]: top_left[0] + nose_width] = final_nose

        cv2.imshow("Frame", frame)

        k = cv2.waitKey(1)

        if k % 256 == 27:
            print("escape")
            break
        elif k % 256 == 32:
            img_name = "screenshot_{}.png".format(str(uuid.uuid4())[:8])
            cv2.imwrite(img_name, frame)
            print("screenshot taken")
            img_counter += 1
            cv2.waitKey(100)

    cap.release()
    cv2.destroyAllWindows()

elif choice == '3':


    cap = cv2.VideoCapture(0)
    # VideoCapture sınıfının başlangıcında belirli bir frame sayısını ayarla

    horns_image = cv2.imread("devil_horns.png")
    _, frame = cap.read()
    rows, cols, _ = frame.shape
    horns_mask = np.zeros((rows, cols), np.uint8)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    cv2.namedWindow("Devil Horns App")
    img_counter = 0

    while True:
        _, frame = cap.read()
        horns_mask.fill(0)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(frame)
        for face in faces:
            landmarks = predictor(gray_frame, face)

            # Ortak noktalar
            top_of_head = (landmarks.part(27).x, landmarks.part(0).y)
            left_horn = (landmarks.part(8).x, landmarks.part(1).y)
            right_horn = (landmarks.part(16).x, landmarks.part(6).y)

            horn_width = int(hypot(left_horn[0] - right_horn[0], left_horn[1] - right_horn[1]) * 1.2)
            horn_height = int(horn_width * 0.7)

            # Boynuz boyutlarını iki katına çıkar
            new_width = int(horn_width * 2)
            new_height = int(horn_height * 2)

            # Boynuz resmini ölçekle
            scaled_horns = cv2.resize(horns_image, (new_width, new_height))

            # Boynuz maskesini de ölçekle
            scaled_horns_gray = cv2.cvtColor(scaled_horns, cv2.COLOR_BGR2GRAY)
            _, scaled_horns_mask = cv2.threshold(scaled_horns_gray, 25, 255, cv2.THRESH_BINARY_INV)

            top_left = (int(top_of_head[0] - new_width / 2),
                        int(top_of_head[1] - new_height))
            bottom_right = (top_left[0] + new_width, top_left[1] + new_height)

            # Yeni boyutlarda boynuzları yerleştir
            scaled_horns_area = frame[top_left[1]: bottom_right[1], top_left[0]: top_left[0] + new_width]
            scaled_horns_area_no_horns = cv2.bitwise_and(scaled_horns_area, scaled_horns_area, mask=scaled_horns_mask)
            final_scaled_horns = cv2.add(scaled_horns_area_no_horns, scaled_horns)

            frame[top_left[1]: bottom_right[1], top_left[0]: top_left[0] + new_width] = final_scaled_horns

        cv2.imshow("Devil Horns App", frame)

        k = cv2.waitKey(1)

        if k % 256 == 27:
            print("escape")
            break
        elif k % 256 == 32:
            img_name = "screenshot_{}.png".format(str(uuid.uuid4())[:8])
            cv2.imwrite(img_name, frame)
            print("screenshot taken")
            img_counter += 1
            cv2.waitKey(100)

    cap.release()
    cv2.destroyAllWindows()







elif choice == '4':
    # Mustache functionality
    cap = cv2.VideoCapture(0)
    mustache_image = cv2.imread("mustache_yellow_.png")  # Change the file extension if needed
    _, frame = cap.read()
    rows, cols, _ = frame.shape
    mustache_mask = np.zeros((rows, cols), np.uint8)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    cv2.namedWindow("Mustache App")
    img_counter = 0

    while True:
        _, frame = cap.read()
        mustache_mask.fill(0)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(frame)
        for face in faces:
            landmarks = predictor(gray_frame, face)

            # Mustache coordinates
            top_mustache = (landmarks.part(49).x, landmarks.part(49).y)
            center_mustache = (landmarks.part(51).x, landmarks.part(51).y)
            left_mustache = (landmarks.part(48).x, landmarks.part(48).y)
            right_mustache = (landmarks.part(54).x, landmarks.part(54).y)

            mustache_width = int(hypot(left_mustache[0] - right_mustache[0],
                                   left_mustache[1] - right_mustache[1]) * 1.7)
            mustache_height = int(mustache_width * 0.5)

            # New mustache position
            offset = int(10 * mustache_height / 100)  # 1 cm offset (adjust as needed)
            top_left = (int(center_mustache[0] - mustache_width / 2),
                        int(center_mustache[1] - mustache_height / 2) - offset)
            bottom_right = (int(center_mustache[0] + mustache_width / 2),
                            int(center_mustache[1] + mustache_height / 2) - offset)

            # Adding the new mustache
            mustache = cv2.resize(mustache_image, (mustache_width, mustache_height))
            mustache_gray = cv2.cvtColor(mustache, cv2.COLOR_BGR2GRAY)
            _, mustache_mask = cv2.threshold(mustache_gray, 25, 255, cv2.THRESH_BINARY_INV)

            mustache_area = frame[top_left[1]: top_left[1] + mustache_height,
                                  top_left[0]: top_left[0] + mustache_width]
            mustache_area_no_mustache = cv2.bitwise_and(mustache_area, mustache_area, mask=mustache_mask)
            final_mustache = cv2.add(mustache_area_no_mustache, mustache)

            frame[top_left[1]: top_left[1] + mustache_height,
                  top_left[0]: top_left[0] + mustache_width] = final_mustache

        cv2.imshow("Mustache App", frame)

        k = cv2.waitKey(1)

        if k % 256 == 27:
            print("Escape: Exiting the application.")
            break
        elif k % 256 == 32:
            img_name = "screenshot_{}.png".format(str(uuid.uuid4())[:8])
            cv2.imwrite(img_name, frame)
            print("Spacebar: Screenshot taken")
            img_counter += 1
            cv2.waitKey(100)

    cap.release()
    cv2.destroyAllWindows()


elif choice == '0':
    print("0: Exiting the application.")

else:
    print("Invalid choice. Please enter 1, 2, 3, 4, or 0.")
