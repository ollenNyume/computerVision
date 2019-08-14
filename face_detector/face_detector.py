import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img = cv2.imread("wed.jpg")  # load image
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale

# search for classifier and return coordinates of face eg upper lower points to make a square
faces = face_cascade.detectMultiScale(grey_img, scaleFactor=1.1, minNeighbors=5)  # smaller value = more accuracy
print(type(faces))
print(faces)

# draw rectangle
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (100, 190, 200), 3)

resized_img= cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))

# displaying image
cv2.imshow("gray",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
