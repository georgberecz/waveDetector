import numpy as np
import threading
import time
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

#overlapThreshold is a precentage in range 0-1
def mergeRectangles(rects, overlapThres):
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
                                time.sleep(2)
                                break;
                if (not overlap):
                        rects2.append((x, y, w, h))
        return rects2

videoFileName="VIDEO0062.mp4"
capture = cv2.VideoCapture(videoFileName)

faceCascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
i = 0;
factor = 0.5
de_factor = 1 / factor

ret, frame = capture.read()

#starting the first frame threads
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
resized_gray = cv2.resize(gray, (0,0), fx = factor, fy = factor)
t = Thread(target=detectFaces, args=(resized_gray,faceCascade))
t.start()


while (ret != False):
	# Capture frame-by-frame
	time.sleep(0.2)
	if i % 1 == 0:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		resized_gray = cv2.resize(gray, (0,0), fx = factor, fy = factor)
		
		if t.is_alive():
			print "Thread still running"
		else:
			t = Thread(target=detectFaces, args=(resized_gray,faceCascade))
			t.start()
			#print "Thread finished -> restart"	
	#show the frame

	rects_lock.acquire()
	rects = mergeRectangles(rects, 0.5)
	
	for rect in rects:
		cv2.rectangle(frame,(int(rect[0] * de_factor), int(rect[1] * de_factor)),(int(rect[0] * de_factor + rect[2] * de_factor), int(rect[1] * de_factor + rect[3] * de_factor)),(255,0,0),3)
	rects_lock.release()

	cv2.imshow('frame',frame)
	i += 1

	ret, frame=capture.read()
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()