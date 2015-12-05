import cv2

class FaceDetector:

	def __init__(self, cascade_path):
		self.cascade_path = cascade_path
		self.face_cascade = cv2.CascadeClassifier(cascade_path)

	def detect(self, frame):
		# self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		return self.face_cascade.detectMultiScale(frame, scaleFactor = 1.2, minNeighbors = 5, minSize = (100, 100))