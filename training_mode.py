import cv2
import numpy as np
import os
from PIL import Image

class TrainingMode:

	FRAMECAPTURE_EXTENSION = ".png"
	ALREADY_TRAINED = False 			# Variable to ignore keystroke events from facerec.py when training was already started
	FACES_ROOT_PATH = "faces"
	FACE_PATH = ""
	MODEL_PATH = None 	# for existing models
	MODEL_NAME = None 	# for newly created models
	EXIT_FLAG = False
	IMAGE_COUNT = 50

	def __init__(self, person_name, existing_model, new_model_name):
		print "Entering training mode. Make sure that only the desired training face is visible to the camera."

		if existing_model is None and new_model_name is not None:
			print "No model given, creating new one with name '" + new_model_name + "'."
			self.MODEL_NAME = new_model_name

		elif existing_model is not None and new_model_name is None:
			print "Model given at '" + model + "', updating with new date."
			self.MODEL_PATH = model
		else:
			print "Wrong arguments, quitting"
			return

		self.image_count = 0
		self.setup_directory_structure()

	# Creates directory structure for training faces like the following example:
	# /project_root/faces/1/[0-9].png
	# /project_root/faces/2/[0-9].png
	# and so on
	def setup_directory_structure(self):
		self.FACES_ROOT_PATH = "faces"
		self.FACE_PATH = ""

		if not os.path.exists(self.FACES_ROOT_PATH):
			os.makedirs(self.FACES_ROOT_PATH)

		i = 1
		while True:
			if not os.path.exists(self.FACES_ROOT_PATH + "/" + str(i)):

				# TODO: The labels for OpenCV need to be numeric values. Use the label for person's names and generate a number out of it?
				self.face_label = str(i)
				self.FACE_PATH = self.FACES_ROOT_PATH + "/" + str(i)
				os.makedirs(self.FACE_PATH)
				break
			i += 1

	def consume(self, frame, ROIs, ROIs_coordinates, faces):
		if len(ROIs) == 1:
			ROI_x = ROIs_coordinates[0][0]
			ROI_y = ROIs_coordinates[0][1]
			ROI_w = ROIs[0].shape[0]
			ROI_h = ROIs[0].shape[1]
			cv2.rectangle(frame, (ROI_x, ROI_y), (ROI_x+ROI_w, ROI_y+ROI_h), (0, 0, 255))

		if not cv2.waitKey(1) & 0xFF == ord('p'):
			return frame

		if self.image_count == self.IMAGE_COUNT:
			if self.ALREADY_TRAINED:
				return

			print str(self.image_count) + " images were captured. Training recognizer ..."
			images, labels = self.get_images_and_labels(self.FACE_PATH)
			recognizer = cv2.createLBPHFaceRecognizer()
			self.train_recognizer(recognizer, images, labels)
			self.EXIT_FLAG = True
			return

		if len(faces) > 1:
			print "Detected more than one face. Skipping frame ..."
			return None

		if len(faces) == 0:
			print "No faces in the image. Skipping frame ..."
			return None

		self.image_count += 1
		cv2.imwrite(self.FACE_PATH + "/" + str(self.image_count) + self.FRAMECAPTURE_EXTENSION, ROIs[0])
		print "Saved image!"

		return frame


	def get_images_and_labels(self, path):
		# image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]
		image_paths = [f for f in os.listdir(path) if f.endswith('.png')]
		labels = []
		images = []

		for image_path in image_paths:
			image_pil = Image.open(self.FACE_PATH + "/" + image_path).convert("L")		# No need for grayscale conversion, already done by face_detector.py
			image = np.array(image_pil, "uint8")	# Convert to NumPy array
			images.append(image)
			# Every image needs to have a seperate reference to its label
			labels.append(int(self.face_label))

		return images, labels

	def train_recognizer(self, recognizer, images, labels):
		if self.MODEL_PATH is None:
			recognizer.train(images, np.array(labels))
			recognizer.save(self.MODEL_NAME)
			print "Training done"
			return

		recognizer.load(self.MODEL_PATH)
		recognizer.train(images, np.array(labels))
		recognizer.save(self.MODEL_PATH)
		print "Training done, model updated"


