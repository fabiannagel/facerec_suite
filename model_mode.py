import cv2

class ModelMode:
	MODEL_PATH = None
	EXIT_FLAG = False

	def __init__(self, model):
		print "Entering recognition mode with model '" + model + "'"
		self.MODEL_PATH = model
		self.recognizer = cv2.createLBPHFaceRecognizer()
		self.recognizer.load(model)

	def consume(self, frame, ROIs, ROIs_coordinates, faces):
		if len(faces) == 0:
			return frame

		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)		# TODO: Move this to facerec.py

		predicted_label, conf = self.recognizer.predict(gray_frame)

		if len(ROIs) == 1:
			ROI_x = ROIs_coordinates[0][0]
			ROI_y = ROIs_coordinates[0][1]
			ROI_w = ROIs[0].shape[0]
			ROI_h = ROIs[0].shape[1]
			cv2.rectangle(frame, (ROI_x, ROI_y), (ROI_x+ROI_w, ROI_y+ROI_h), (0, 0, 255))
			
			if conf < 100:
				cv2.putText(frame, str(predicted_label), (ROI_x, ROI_y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

			print "Predicted label: " + str(predicted_label) + " (confidency " + str(conf) + ")"

		return frame






	