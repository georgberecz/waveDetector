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
	lower_red_first = np.array([169,25,5])
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

def draw_rects(frame, rects):
	for rect in rects:
		if rect is not None:
			cv2.rectangle(frame,(int(rect[0] * de_factor), int(rect[1] * de_factor)),(int(rect[0] * de_factor + rect[2] * de_factor), int(rect[1] * de_factor + rect[3] * de_factor)),(255,0,0),3)

#It matches the current frame with the previous one and adds Null if no match was found
def match_rects(rects, prev_frames_rects):
	output = []
	prev_rects = []
	rects2 = copy.deepcopy(rects)
	
	#take all previous frame rectangles
	if len(prev_frames_rects) > 0:
		prev_rects = prev_frames_rects[-1]

	#match all previous frame rectangles
	for j in range(len(prev_rects)):
		
		#if (j >= len(prev_rects)): continue; #somehow there appears a bug where j actually can go bigger than len(prev_rects)
		
		if (prev_rects[j] is None):
			output.append(None)
			continue
		prev_x = prev_rects[j][0]
		mindist = 9999
		minrect = None
		if (len(rects2) > 0):
			minrect = rects2[0]
		for r in rects2:
			if (r is not None):
				dist = abs(r[0] - prev_x)
				if (dist < mindist):
					mindist = dist
					minrect = r
		if (mindist > 50):
			output.append(None);
		else:
			rects2.remove(minrect)
			output.append(minrect)
	for r in rects2:
		output.append(r)
	
	return output

def smooth_rects(rects, prev_frames_rects, steps):
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
	return rects

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
			continue;
	#add missing frames
	#for (i in range(start, end-1)):
	return corrected_rects


videoFileName="VIDEO0062.mp4"
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
	gesture_detected = False

	# Capture frame-by-frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	resized_gray = cv2.resize(gray, (0,0), fx = factor, fy = factor)
	
	if not t.is_alive():
		#Process rectangles
		rects_lock.acquire()
		rects = match_rects(rects, prev_frames_rects)
		rects = smooth_rects(rects, prev_frames_rects, 5)
		prev_frames_rects.append(rects)
		corrected_prev_frames_rects.append(rects)
		correct_rects(corrected_prev_frames_rects, prev_frames_rects, 5)
		
		print_rects(rects)
		rects_lock.release()
		#Start new thread
		t = Thread(target=detectFaces, args=(resized_gray,faceCascade))
		t.start()
		#print "Thread finished -> restart"
#show the frame

	rects_lock.acquire()
	if (prev_frame is not None):
		draw_rects(prev_frame, prev_frames_rects[-2])


	rects_lock.release()
	#print different boxes depending if gesture was detected
	if gesture_detected:
		print_gesture_indicator(frame)
	else:
		print_no_gesture_indicator(frame)

	#show the frame
	cv2.imshow('frame', prev_frame)
	cv2.imshow('frame',frame)
	
	prev_frame = frame
	ret, frame=capture.read()
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()
