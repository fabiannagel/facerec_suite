import cv2, os
import numpy as np
import argparse

from face_detector import FaceDetector
from training_mode import TrainingMode
from model_mode import ModelMode

def draw_bounding_box(faces):
	print "foo"

ap = argparse.ArgumentParser()

group = ap.add_mutually_exclusive_group(required=True)
group.add_argument("--train", help="Create a new face recognition model")
group.add_argument("--update-model", help="Update an existing face recognition model")
group.add_argument("--model", help="Path for existing face recognition model")
group.add_argument("--clear", help="Clear all data (models and images)") # TODO

args = vars(ap.parse_args())

frame_class = None 	# Class to which every grabbed frame will be delivered

if args["train"]:
	new_model_name = args["train"]
	frame_class = TrainingMode(None, new_model_name)

if args["update_model"]:
	model_path = args["update_model"]
	# TODO: Check if file exists
	frame_class = TrainingMode(model_path, None)

if args["model"]:
	model_path = args["model"]	# Model file
	# TODO: Check if file exists
	frame_class = ModelMode(model_path)

# Face detection is needed in all modes
face_detector = FaceDetector("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while(True):	# Main loop
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx = 0.3, fy = 0.3)

    faces = face_detector.detect(frame)
    for (x, y, w, h) in faces:
		ROI = frame[y: y+h, x: x+w]		 	# Face region
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('p'):
    	if frame_class.consume(ROI, faces):
    		break

cap.release()
cv2.destroyAllWindows()
