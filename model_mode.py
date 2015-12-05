import cv2

class ModelMode:

	def __init__(self, model):
		self.model = model
		print "Entering recognition mode with model '" + model + "'"
	