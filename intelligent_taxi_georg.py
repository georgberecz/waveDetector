import numpy as np
import threading
from threading import Thread
import time
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

videoFileName="VIDEO0061.mp4"
capture = cv2.VideoCapture(videoFileName)

faceCascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
i = 0;
factor = 0.5
de_factor = 1 / factor
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))


ret, frame = capture.read()

while (ret != False):
	gesture_detected = False

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
	#gray = cv2.blur(gray,(5,5))
	#cv2.imshow('gray',gray)
	#print rectangles
	rects_lock.acquire()
	for rect in rects:
		x1 = int(rect[0] * de_factor)
		x2 = int(rect[0] * de_factor + rect[2] * de_factor)
		y1 = int(rect[1] * de_factor)
		y2 = int(rect[1] * de_factor + rect[3] * de_factor)
		y_difference = int((y2 - y1) / 3)
		cv2.rectangle(frame,
					  (x1,y1),(x2,y2),
					  (255,0,0),
					  1)

		ret,thresh = cv2.threshold(gray[y1:(y1+y_difference),x1:x2],
								   110,
								   255,
								   cv2.THRESH_BINARY)
								   
		if thresh is not None:
			rec_human = frame[y1:(y1+y_difference),x1:x2]
			thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
			skin_color =  cv2.mean(rec_human,thresh)
			
			skin_min_threshold = tuple(np.subtract(skin_color, (20, 20, 20, 0)))
			skin_max_threshold = tuple(np.add(skin_color, (20, 20, 20, 0)))
			skin_mask = cv2.inRange(rec_human,skin_min_threshold,skin_max_threshold)
			skin_mask = cv2.dilate(skin_mask,kernel,iterations = 7)

			cv2.imshow('skin_mask',skin_mask)
			#cv2.imshow('thresh',thresh)
			contours, hierarchy = cv2.findContours(skin_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,offset=(x1,y1))
			accepted_contours = 0
			for contour in contours:
				area = cv2.contourArea(contour)
				if area > 150:
					accepted_contours += 1
			if accepted_contours >= 2:
				gesture_detected = True
			#cv2.drawContours(frame, contours, -1, (0,255,0), -1)

			

	rects_lock.release()


	#print different boxes depending if gesture was detected
	if gesture_detected:
		print_gesture_indicator(frame)
	else:
		print_no_gesture_indicator(frame)

	#show the frame
	cv2.imshow('frame',frame)
	i += 1

	ret, frame=capture.read()
	#time.sleep(1)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()
