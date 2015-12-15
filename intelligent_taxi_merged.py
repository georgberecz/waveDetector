import numpy as np
import threading
import time
from threading import Thread
import cv2
import copy

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
	rects = merge_rects(rects, 0.5)
	rects_lock.release()
	return True

def calculate_hsv_mask (image):
	lower_red_first = np.array([170,25,5])
	upper_red_first = np.array([179,150,250])
	lower_red_second = np.array([0,25,5])
	upper_red_second = np.array([15,150,250])
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
							 np.array([0,lower_bound[1],lower_bound[2]]),upper_bound)
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

def draw_rects(frame, rects):
	for rect in rects:
		if rect is not None:
			cv2.rectangle(frame,(int(rect[0] * de_factor), int(rect[1] * de_factor)),(int(rect[0] * de_factor + rect[2] * de_factor), int(rect[1] * de_factor + rect[3] * de_factor)),(255,0,0), 3)

#It matches the current frame with the previous one and adds Null if no match was found
def match_rects(rects, corrected_prev_frames_rects, prev_frames_rects, steps):
	output = []
	corrected_output = []
	prev_rects = []
	rects2 = copy.deepcopy(rects)

	if len(prev_frames_rects) > 0:
		prev_rects = prev_frames_rects[-1]

	#match all previous frame rectangles
	for j in range(len(prev_rects)):
		check_frame_num = 1
		check_rects = prev_rects

		if (prev_rects[j] is None):
			#seek earlier rects
			found = False
			while (check_frame_num >= 0 and check_frame_num < steps+1):
				check_frame_num += 1
				if (prev_frames_rects[-check_frame_num][j] != None):
					found = True
					check_rects = prev_frames_rects[-check_frame_num]
					break
			if (not found):
				output.append(None)
				corrected_output.append(None)
				# (1) this place is certainly dead, no previous rectangle within step
				continue

		mindist, minrect = findMinRect(check_rects[j], rects2)

		if (mindist > 50):
			output.append(None)
			#The current frame doesn't contain any rects but we'll still add whatever rectangle we had previously
			#because it has to be in the range of steps otherwise we would have exit at [1]
			corrected_output.append(check_rects[j])
		else:
			rects2.remove(minrect)
			corrected_output.append(minrect)
			if (check_frame_num > 1):
				output.append(minrect)
			else:
				output.append(None)
	for r in rects2:
		output.append(r)
		corrected_output.append(r)

	prev_frames_rects.append(output)
	corrected_prev_frames_rects.append(corrected_output)

def findMinRect(rect, rect_list):
	prev_x = rect[0]
	mindist = 9999
	minrect = None
	if (len(rect_list) > 0):
		minrect = rect_list[0]
	for r in rect_list:
		if (r is not None):
			dist = abs(r[0] - prev_x)
			if (dist < mindist):
				mindist = dist
				minrect = r
	return mindist, minrect

def smooth_rects(corrected_prev_frames_rects, prev_frames_rects, steps):

	rects = prev_frames_rects[-1]
	prev_len = len(prev_frames_rects)
	for i in range(len(rects)):
		if rects[i] is None:
			continue;
		#if (len(rects)) > 1: time.sleep(1)
		(x, y, w, h) = rects[i]
		x2 = x+w
		y2 = y+h
		previous = prev_frames_rects[max(0, prev_len-steps) : prev_len]
		for j in range(len(previous)):
			#find the corresponding rectangle in rectangle list
			if len(previous[j]) <= i:
				continue;
			minrect = previous[j][i]
			if minrect is None:
				continue;
			#edit chosen rectangle
			(px, py, pw, ph) = minrect
			px2 = px + pw
			py2 = py + ph
			x = slerp(x, px, 0.5 / (j+1))
			y = slerp(y, py, 0.5 / (j+1))
			x2 = slerp(x2, px2, 0.5 / (j+1))
			y2 = slerp(y2, py2, 0.5 / (j+1))
		rects[i] = (x, y, x2-x, y2-y)
	prev_frames_rects[-1] = rects

def print_rects(rects):
	result = "["
	for i in range(len(rects)):
		if (rects[i] != None):
			result += "A"
		else:
			result += "x"
		if (i < len(rects)-1):
			result += " "
	result += "]"
	print result;


def slerp(a, b, value):
	return b*value + a*(1-value)

#overlapThreshold is a precentage in range 0-1
def merge_rects(rects, overlapThres):
	rects2 = []
	for i in range(len(rects)):
		overlap = False
		(x, y, w, h) = rects[i]
		for j in range(i+1, len(rects)):
			(x2, y2, w2, h2) = rects[j]
			ileft = max(x, x2)
			iright = min(x+w, x2+w2)
			itop = max(y, y2)
			ibottom = min(y+h, y2+h2)
			si = max(0, iright - ileft) * max(0, ibottom - itop)
			sa = (w * h)
			sb = (w2 * h2)
			if (si / sa > overlapThres or si / sb > overlapThres):
				print "OVERLAP"
				nx1 = min(x, x2)
				ny1 = min(y, y2)
				nx2 = max(x + w, x2 + w2)
				ny2 = max(y + h, y2 + h2)
				rects[j] = (nx1, ny1, nx2-nx1, ny2-ny1)
				overlap = True;
				#time.sleep(2)
				break;
		if (not overlap):
			rects2.append((x, y, w, h))
	return rects2

def remove_false_positive_rects(corrected_rects, rects):
	if len(corrected_rects) < 3:
		return

	for i in range(len(corrected_rects[-1])):
		if (len(corrected_rects[-2]) <= i or len(corrected_rects[-3]) <= i): continue
		if (corrected_rects[-1][i] is None and corrected_rects[-2][i] is not None and corrected_rects[-3][i] is None):
			corrected_rects[-2][i] = None
			rects[-2][i] = None


def correct_rects(corrected_rects, rects, step):
	if len(rects) < 2:
		return corrected_rects;



	start = len(rects)-step
	if start < 0:
		start = 0;
	end = len(rects)

	curr_rects = rects[-1]

	for i in range(len(curr_rects)):
		if (curr_rects[i] == None):
			predecessor = None
			for j in range(end-1, start, -1):
				if (len(rects[j]) > i and rects[j][i] != None):
					predecessor = rects[j][i];
					break;
			if predecessor != None:
				corrected_rects[-1][i] = predecessor

	#add missing frames
	#for(i in range(start, end-1)):
	return corrected_rects


##########################################################################
# START
##########################################################################

videoFileName="VIDEO0061.mp4"
capture = cv2.VideoCapture(videoFileName)

faceCascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
# use factor depending on video size
factor = 0.5
de_factor = 1 / factor

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))

ret, frame = capture.read()

#starting the first frame threads
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
resized_gray = cv2.resize(gray, (0,0), fx = factor, fy = factor)
t = Thread(target=detectFaces, args=(resized_gray,faceCascade))
t.start()

prev_frames_rects = []
corrected_prev_frames_rects = []
prev_frame = None
while (ret != False):
	#time.sleep(0.2)
	gesture_detected = False

	# Capture frame-by-frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	resized_gray = cv2.resize(gray, (0,0), fx = factor, fy = factor)
	
	if not t.is_alive():
		#Process rectangles
		rects_lock.acquire()
		match_rects(rects, corrected_prev_frames_rects, prev_frames_rects, 5)
		#smooth_rects(corrected_prev_frames_rects, prev_frames_rects, 5)
		#correct_rects(corrected_prev_frames_rects, prev_frames_rects, 5)
		remove_false_positive_rects(corrected_prev_frames_rects, prev_frames_rects)
		
		print_rects(corrected_prev_frames_rects[-1])
		rects_lock.release()
		#Start new thread
		t = Thread(target=detectFaces, args=(resized_gray,faceCascade))
		t.start()
		#print "Thread finished -> restart"

	final_rects = []
	if (len(corrected_prev_frames_rects) > 1):
		final_rects = corrected_prev_frames_rects[-2]

	for rect in final_rects:
		if rect is not None:
			x1 = int(rect[0] * de_factor)
			x2 = int(rect[0] * de_factor + rect[2] * de_factor)
			y1 = int(rect[1] * de_factor)
			y2 = int(rect[1] * de_factor + rect[3] * de_factor)
			
			# increase size of rectangle by factor (to make sure arm is inside the box)
			inc_factor = 0.5
			x_difference = x2 - x1
			x1 = int (x1 - x_difference * inc_factor)
			if x1 < 0: x1 = 0
			x2 = int (x2 + x_difference * inc_factor)
			
			# y_difference defines the upper third of the recognized human
			y_difference = int((y2 - y1) / 3)
			y1 = int (y1 - y_difference * inc_factor)
			if y1 < 0: y = 0
			y_difference += int(y_difference * inc_factor)
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

			#convert to hsv
			hsv_frame = cv2.cvtColor(frame[y1:(y1+y_difference),x1:x2], cv2.COLOR_BGR2HSV)
			hsv_frame_blurred = cv2.blur(hsv_frame,(3,3))
			hsv_mask = calculate_hsv_mask(hsv_frame)
			
			thresh = hsv_mask
			thresh = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel, iterations = 2)
			skin_color_thresh = cv2.erode(thresh,kernel,iterations = 2)
			thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 3)
					  
			if thresh is not None:
				rec_human = frame[y1:(y1+y_difference),x1:x2]
				skin_color =  cv2.mean(rec_human,skin_color_thresh)
				hsv_skin_color = cv2.cvtColor(np.uint8([[skin_color]]), cv2.COLOR_BGR2HSV)
				skin_mask = calculate_skin_mask(hsv_skin_color, hsv_frame_blurred)
				skin_mask = cv2.dilate(skin_mask,kernel,iterations = 1)

				end_mask = thresh & skin_mask
				kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
				end_mask = cv2.dilate(end_mask,kernel,iterations = 3)

				end_mask = cv2.morphologyEx(end_mask, cv2.MORPH_OPEN, kernel, iterations = 2)
				end_mask = cv2.morphologyEx(end_mask, cv2.MORPH_CLOSE, kernel, iterations = 2)


				#cv2.imshow('skin_mask',skin_mask)
				#cv2.imshow('thresh',thresh)
				cv2.imshow('end_mask',end_mask)

				if (cv2.__version__[0] == '3'):
					im2, contours, hierarchy = cv2.findContours(end_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,offset=(x1,y1))
				else:
					contours, hierarchy = cv2.findContours(end_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,offset=(x1,y1))

				accepted_contours = 0
				for contour in contours:
					area = cv2.contourArea(contour)
					if area > 150:
						accepted_contours += 1
				if accepted_contours >= 2:
					gesture_detected = True

	#print different boxes depending if gesture was detected
	if gesture_detected:
		print_gesture_indicator(frame)
	else:
		print_no_gesture_indicator(frame)

	if (prev_frame is not None):
		draw_rects(prev_frame, final_rects)
		#show the frame
		cv2.imshow('frame', prev_frame)


	prev_frame = frame
	ret, frame=capture.read()
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()
