import numpy as np
import cv2


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')
cap = cv2.VideoCapture(0)
while(True):
	# Capture frame-by-frame
	ret,frame=cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = faceCascade.detectMultiScale(gray)
	#show the frame
	for rect in rects:
		cv2.rectangle(frame,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(255,0,0),1)
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()