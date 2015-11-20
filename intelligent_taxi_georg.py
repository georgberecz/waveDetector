import numpy as np
import cv2


videoFileName = "VIDEO0060.mp4"
capture = cv2.VideoCapture(videoFileName)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')
i = 0;
factor = 0.5
de_factor = 1 / factor

ret, frame = capture.read()

print ret

while (ret != False):
	# Capture frame-by-frame
	
	if i % 2 == 0:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		resized_gray = cv2.resize(gray, (0,0), fx = factor, fy = factor)
		rects = faceCascade.detectMultiScale(resized_gray)
	#show the frame
	for rect in rects:
		cv2.rectangle(frame,(int(rect[0] * de_factor), int(rect[1] * de_factor)),(int(rect[0] * de_factor + rect[2] * de_factor), int(rect[1] * de_factor + rect[3] * de_factor)),(255,0,0),1)
	cv2.imshow('frame',frame)
	i += 1

	ret, frame=capture.read()
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()
