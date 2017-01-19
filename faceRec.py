import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

trnImages = []
trnLabels = []

trnImages.append(cv2.cvtColor(cv2.imread("detectedFace0.jpg"), cv2.COLOR_BGR2GRAY))
trnLabels.append(1)
'''
trnImages.append(cv2.cvtColor(cv2.imread("detectedFace1-1.jpg"), cv2.COLOR_BGR2GRAY))
trnLabels.append(2)

trnImages.append(cv2.cvtColor(cv2.imread("detectedFace1-2.jpg"), cv2.COLOR_BGR2GRAY))
trnLabels.append(2)

trnImages.append(cv2.cvtColor(cv2.imread("detectedFace1-3.jpg"), cv2.COLOR_BGR2GRAY))
trnLabels.append(2)

trnImages.append(cv2.cvtColor(cv2.imread("detectedFace1-4.jpg"), cv2.COLOR_BGR2GRAY))
trnLabels.append(2)

trnImages.append(cv2.cvtColor(cv2.imread("detectedFace1-5.jpg"), cv2.COLOR_BGR2GRAY))
trnLabels.append(2)
'''

labelPointer = len(trnImages)

trnImages = np.array(trnImages)
trnLabels = np.array(trnLabels)

#createEigenFaceRecognizer
#createFisherFaceRecognizer
#createLBPHFaceRecognizer

model = cv2.createLBPHFaceRecognizer(threshold=100) # the second argument is the threshold
model.train(trnImages, trnLabels)

while 1:
	try:
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)

		faceDetected = len(faces)>0
		img2 = gray.copy()

		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]

			prediction = model.predict(cv2.resize(roi_gray, (230,300)))
			print prediction
			label = int(prediction[0])
			cv2.putText(img, "Label: %s"%label, (x-2,y-2), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
			
			hMarge = round(h*0.15)
			yStart = y-hMarge if (y-hMarge) >= 0 else y 
			yEnd = y+h+hMarge if (y+h+hMarge) <= img2.shape[0] else y+h

			faceRoi = img2[yStart:yEnd, x:x+w]
			faceRoi = cv2.resize(faceRoi, (230,300))
			
			if label>0: 
				updLabels = np.array([label])
			else: 
				labelPointer += 1
				updLabels = np.array([labelPointer])

			updImages = np.array([faceRoi])
			model.update(updImages, updLabels)
			
			
			eyes = eye_cascade.detectMultiScale(roi_gray)
			for (ex,ey,ew,eh) in eyes:
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
		
		cv2.imshow('img',img)
	except:
		pass

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()