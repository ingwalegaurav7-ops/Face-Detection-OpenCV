import cv2

# Load Haar Cascade
haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Try camera ID 0 first
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("❌ Camera not opening")
    exit()

while True:
    ret, img = cam.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = haar_cascade.detectMultiScale(grayImg, 1.3, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", img)

    key = cv2.waitKey(10) & 0xFF
    if key == 27:   # ESC key
        break

cam.release()
cv2.destroyAllWindows()
