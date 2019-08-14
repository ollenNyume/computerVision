import cv2
import time

first_frame = None
video = cv2.VideoCapture(0)  # trigger built in camera

while True:
    check, frame = video.read()  # read first frame of video
    print(check)
    print(frame)

    greyscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    greyscale_img_blr = cv2.GaussianBlur(greyscale_img, (21, 21), 0)  # blur image to remove noise

    if first_frame is None:
        first_frame = greyscale_img
        continue  # next iteration

    # calculate difference between current frame and first frame
    delta_frame = cv2.absdiff(first_frame, greyscale_img_blr)
    # assigning thresholds
    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta, None, iterations=2)  # smoothen threshold image

    # finding and storing contours into cnts variable
    (_, cnts, _) = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # keep some contours > 1000 px
    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
    # draw rectangle around bigger contours
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
    # time.sleep(3)  # script waits
    greyscale_img = cv2.resize(greyscale_img, (int(greyscale_img.shape[1]/2), int(greyscale_img.shape[0]/2)))
    cv2.imshow("original", greyscale_img)  # show first frame in window
    delta_frame = cv2.resize(delta_frame, (int(delta_frame.shape[1]/2), int(delta_frame.shape[0]/2)))
    cv2.imshow("delta", delta_frame)
    thresh_delta = cv2.resize(thresh_delta, (int(thresh_delta.shape[1]/2), int(thresh_delta.shape[0]/2)))
    cv2.imshow("threshold", thresh_delta)
    frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
    cv2.imshow("color frame", frame)

    key = cv2.waitKey(1)  # print new frame every 1 millisecond
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
