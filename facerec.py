import cv2, os
import numpy as np
import argparse
import sys

from face_detector import FaceDetector
from training_mode import TrainingMode
from model_mode import ModelMode

ROI = None

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

	person_name = None
	while person_name is None or person_name == "":
		person_name = raw_input("Please enter the person's name: ")


	frame_class = TrainingMode(person_name, None, new_model_name)

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
    frame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)

    ROIs = []					# Array for segmeted faces
    ROIs_coordinates = []		# Array for x,y coordinates of segmented faces to draw bounding boxes

    faces = face_detector.detect(frame)
    for (x, y, w, h) in faces:
		ROIs.append(frame[y: y+h, x: x+w])		 	
		ROIs_coordinates.append((x, y))
		# cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

    frame = frame_class.consume(frame, ROIs, ROIs_coordinates, faces)
    if frame_class.EXIT_FLAG:
    	break

    cv2.imshow('frame', frame)


cap.release()
cv2.destroyAllWindows()
