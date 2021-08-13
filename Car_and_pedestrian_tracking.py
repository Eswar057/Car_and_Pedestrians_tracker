import cv2 

img_file = 'car_image.jpg'
video = cv2.VideoCapture('video.mp4')
#video = cv2.VideoCapture(0)

#car
car_tracker_file = 'cars.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

while True:

    (read_successful, frame) = video.read()

    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    else:
        break  

    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    #cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    #pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow("Self Driving Car", frame)

    key = cv2.waitKey(1)

    if key == 81 or key == 113 or key == 13 or key == 9:
        break    

video.release()

print("Code Completed")

"""
    #for image

    img = cv2.imread(img_file)

    black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
"""  