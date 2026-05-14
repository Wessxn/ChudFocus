import cv2 as cv 

video = cv.VideoCapture('pictures/reasl 2.mp4')
while True:
    isTrue, frame = video.read()
    cv.imshow('video', frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()