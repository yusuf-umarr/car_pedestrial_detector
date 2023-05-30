import cv2

img_file = 'cars_image.webp'
video_file = cv2.VideoCapture("moving_cars.mp4")
# video_file = cv2.VideoCapture("Tesla_FSD_avoids_tumbleweed.mp4")
# video_file = cv2.VideoCapture("Cyclist_Hits_Pedestrian.mp4")


# #create car classifier
car_tracker = cv2.CascadeClassifier("cars.xml")
pedestrian_tracker = cv2.CascadeClassifier("haarcascade_fullbody.xml")


# choose an image to detect faces in

img = cv2.imread(img_file)

while True:
    #read the current frame
    (read_successful, frame) = video_file.read()

    if read_successful:
        #convert video to gray
        grayScaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # #detect cars and pedetrians
    cars = car_tracker.detectMultiScale(grayScaled_frame)
    pedestrian = pedestrian_tracker.detectMultiScale(grayScaled_frame)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y, w, h) in pedestrian:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)






    cv2.imshow('Yusuf Car & Pedestfrian Detector', frame)
    key=cv2.waitKey(1)


        #Stop if Q key is pressed
    if key==81 or key== 113:
        break

    #Release the video capture object
video_file.release()

print("code completed!!!")




