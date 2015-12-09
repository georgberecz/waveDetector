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

def calculate_hsv_mask (image):
	lower_red_first = np.array([165,0,10])
	upper_red_first = np.array([179,150,255])
	lower_red_second = np.array([0,0,10])
	upper_red_second = np.array([15,150,255])
	mask_red1 = cv2.inRange(image, lower_red_second, upper_red_second)
	mask_red2 = cv2.inRange(image, lower_red_first, upper_red_first)
	mask = mask_red1 | mask_red2

	return mask

def calculate_skin_mask (skin_color, image):
	lower_bound = np.subtract(skin_color,(2,130,40))
	lower_bound = lower_bound[0][0]
	upper_bound = np.add(skin_color,(4,130,150))
	upper_bound = upper_bound[0][0]
	for i in range(1,3):
		if lower_bound[i] < 0:
			lower_bound[i] = 0
		if upper_bound[i] > 255:
			upper_bound[i] = 255
	if lower_bound[0] < 0:
		mask_1 = cv2.inRange(image,
							 np.array([180+lower_bound[0],lower_bound[1],lower_bound[2]]),
							 np.array([179,upper_bound[1],upper_bound[2]]))
		mask_2 = cv2.inRange(image,
						  np.array([0,lower_bound[1],lower_bound[2]]),
						  upper_bound)
		return (mask_1 | mask_2)
	elif upper_bound[0] > 179:
		mask_1 = cv2.inRange(image,
							 lower_bound,
							 np.array([179,upper_bound[1],upper_bound[2]]))
		mask_2 = cv2.inRange(image,
						 np.array([0,lower_bound[1],lower_bound[2]]),
						 np.array([upper_bound[0] - 180,upper_bound[1],upper_bound[2]]))
		return (mask_1 | mask_2)
	else:
		mask_1 = cv2.inRange(image,lower_bound,upper_bound)
		return mask_1


videoFileName="VIDEO0060.mp4"
capture = cv2.VideoCapture(videoFileName)

faceCascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
i = 0;
# use factor depending on video size
factor = 0.5
de_factor = 1 / factor

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))


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
		# y_difference defines the upper third of the recognized human
		y_difference = int((y2 - y1) / 3)
		cv2.rectangle(frame,
					  (x1,y1),(x2,y2),
					  (255,0,0),
					  1)
		
		hsv_frame = cv2.cvtColor(frame[y1:(y1+y_difference),x1:x2], cv2.COLOR_BGR2HSV)
		hsv_frame_blurred = cv2.blur(hsv_frame,(3,3))
		hsv_mask = calculate_hsv_mask(hsv_frame)
		#ret,thresh = cv2.threshold(gray[y1:(y1+y_difference),x1:x2], 110, 255, cv2.THRESH_BINARY)
		#thresh = hsv_mask
		thresh = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel, iterations = 2)
		skin_color_thresh = cv2.erode(thresh,kernel,iterations = 2)
		#thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 4)

		if thresh is not None:
			rec_human = frame[y1:(y1+y_difference),x1:x2]
			#rec_human_blurred = cv2.blur(rec_human,(3,3))
			skin_color =  cv2.mean(rec_human,skin_color_thresh)
			hsv_skin_color = cv2.cvtColor(np.uint8([[skin_color]]), cv2.COLOR_BGR2HSV)
			skin_mask = calculate_skin_mask(hsv_skin_color, hsv_frame_blurred)
			#thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 4)
			#skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations = 4)
			#skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations = 3)
			#thresh = cv2.dilate(thresh,kernel,iterations = 3)
			#thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 3)
			#skin_color =  cv2.mean(rec_human,thresh)

			#skin_min_threshold = tuple(np.subtract(skin_color, (20, 20, 20, 0)))
			#skin_max_threshold = tuple(np.add(skin_color, (20, 20, 20, 0)))
			#skin_mask = cv2.inRange(rec_human,skin_min_threshold,skin_max_threshold)
			skin_mask = cv2.dilate(skin_mask,kernel,iterations = 1)


			end_mask = skin_mask & thresh

			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
			end_mask = cv2.dilate(end_mask,kernel,iterations = 5)

			#cv2.imshow('skin_mask',skin_mask)
			#cv2.imshow('thresh',thresh)
			#cv2.imshow('end_mask',end_mask)
			contours, hierarchy = cv2.findContours(end_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,offset=(x1,y1))
			#cv2.drawContours(frame, contours, -1, skin_color, -1)
			accepted_contours = 0
			for contour in contours:
				area = cv2.contourArea(contour)
				if area > 150:
					accepted_contours += 1
			if accepted_contours >= 2:
				gesture_detected = True


			

	rects_lock.release()


	#print different boxes depending if gesture was detected
	if gesture_detected:
		print_gesture_indicator(frame)
	else:
		print_no_gesture_indicator(frame)

	#show the frame
	cv2.imshow('frame',frame)
	i += 1
#time.sleep(0.5)
	ret, frame=capture.read()
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()
