import cv2

class ModelMode:
	MODEL_PATH = None

	def __init__(self, model):
		print "Entering recognition mode with model '" + model + "'"
		self.MODEL_PATH = model

	def consume(self, frame, faces):
		print "got a frame"

	