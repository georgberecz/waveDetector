import numpy as np
import threading
from threading import Thread
import cv2

rects = []
rects_lock = threading.Lock()

def print_no_gesture_indicator(image):
	cv2.rectangle(frame,
				  (0,0),
				  (image.shape[1],image.shape[0]),
				  (0,0,255),
				  2)

def print_gesture_indicator(image):
	cv2.rectangle(frame,
				  (0,0),
				  (image.shape[1],image.shape[0]),
				  (0,255,0),
				  2)

def detectFaces(image, faceDetector):
	global rects
	my_rects = faceDetector.detectMultiScale(image)
	rects_lock.acquire()
	rects = my_rects
	rects_lock.release()
	return True

videoFileName="VIDEO0060.mp4"
capture = cv2.VideoCapture(videoFileName)

faceCascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
i = 0;
factor = 0.5
de_factor = 1 / factor

ret, frame = capture.read()

while (ret != False):
	# Capture frame-by-frame
	
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
			print "Thread finished -> restart"
		else:
			print "Thread still running"

	#print rectangles
	rects_lock.acquire()
	for rect in rects:
		cv2.rectangle(frame,
					  (int(rect[0] * de_factor), int(rect[1] * de_factor)),
					  (int(rect[0] * de_factor + rect[2] * de_factor),
					   int(rect[1] * de_factor + rect[3] * de_factor)),
					  (255,0,0),
					  1)
		ret,thresh = cv2.threshold(gray[:,:],
								   100,
								   255,
								   cv2.THRESH_BINARY)
		if thresh is not None:
			print "thresh not None"
			contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			cv2.drawContours(frame, contours, -1, (0,255,0), 1)
	rects_lock.release()


	gesture_detected = False
	#print different boxes depending if gesture was detected
	if gesture_detected:
		print_gesture_indicator(frame)
	else:
		print_no_gesture_indicator(frame)

	#show the frame
	cv2.imshow('frame',frame)
	i += 1

	ret, frame=capture.read()
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()
