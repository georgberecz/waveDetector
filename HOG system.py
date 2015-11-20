import numpy as np
import cv2
#Used sample from https://github.com/Itseez/opencv/blob/master/samples/python2/peopledetect.py


videoFileName = "VIDEO0060.mp4"
capture = cv2.VideoCapture(videoFileName)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')
i = 0;
factor = 0.5
de_factor = 1 / factor

#cv2.HOGDescriptor hog;
#hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

ret, frame = capture.read()

def inside(r, q):
        rx, ry, rw, rh = r
        qx, qy, qw, qh = q
        return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
        for x, y, w, h in rects:
                pad_w, pad_h = int(0.15*w), int(0.05*h)
                cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+y-pad_h), (0, 255, 0), thickness)

while (ret != False):
        # Capture frame-by-frame
        
	found, w = hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)
	#found, w = hog.detectMultiScale(frame, found, 0, Size(6,6), Size(32,32), 1.05, 2)
	found_filtered = []
	for ri, r in enumerate(found):
		for qi, q in enumerate(found):
			if ri != qi and inside(r, q):
				break
			else:
				found_filtered.append(r)
	draw_detections(frame, found)
	draw_detections(frame, found_filtered, 3)
        
     
	cv2.imshow('frame',frame)
	i += 1

	ret, frame=capture.read()
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()
