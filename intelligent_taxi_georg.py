import numpy as np
import threading
from threading import Thread
import cv2

rects = []
rects_lock = threading.Lock()

def detectFaces(image, faceDetector):
	global rects
	my_rects = faceDetector.detectMultiScale(image)
	rects_lock.acquire()
	rects = my_rects
	rects_lock.release()
	return True

videoFileName="VIDEO0062.mp4"
capture = cv2.VideoCapture(videoFileName)

faceCascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
i = 0;
factor = 0.5
de_factor = 1 / factor

while capture.get(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO) < 1:
	# Capture frame-by-frame
	ret,frame=capture.read()
	if i % 1 == 0:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		resized_gray = cv2.resize(gray, (0,0), fx = factor, fy = factor)
		try:
			t.is_alive()
		except NameError:
			t = Thread(target=detectFaces, args=(resized_gray,faceCascade))
			t.start()
		if not t.is_alive():
			t = Thread(target=detectFaces, args=(resized_gray,faceCascade))
			t.start()
	#show the frame

	rects_lock.acquire()
	for rect in rects:
		cv2.rectangle(frame,(int(rect[0] * de_factor), int(rect[1] * de_factor)),(int(rect[0] * de_factor + rect[2] * de_factor), int(rect[1] * de_factor + rect[3] * de_factor)),(255,0,0),1)
	rects_lock.release()

	cv2.imshow('frame',frame)
	i += 1
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()

