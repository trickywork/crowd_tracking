
import cv2
import numpy as np 
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

#DeepSORT -> Importing DeepSORT.
from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet



# Constants.pip install --upgrade pip
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.7

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)


def pre_process(img, net):
	# Create a 4D blob from a frame.
	blob = cv2.dnn.blobFromImage(img, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], True, crop=False)
	# Sets the input to the network.
	net.setInput(blob)
	# Runs the forward pass to get output of the output layers.
	output_layers = net.getUnconnectedOutLayersNames()
	outputs = net.forward(output_layers)
	#(1, 25200, 85) 
	#print(outputs[0].shape)
	return outputs



if __name__ == '__main__':
	# File Initalization
	classesFile = open("classes.txt","r")
	classes = classesFile.read().split('\n')
	modelWeights = "yolov5s.onnx"
	net = cv2.dnn.readNet(modelWeights)
	cap = cv2.VideoCapture('crowd.mp4')
	
	# Deepsort Initalization
	max_cosine_distance = 0.4
	nn_budget = None
	model_filename = './model_data/mars-small128.pb'
	encoder = gdet.create_box_encoder(model_filename, batch_size=1)
	metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
	tracker = Tracker(metric)
	
	# Begin analyze video
	while True:
		success, img = cap.read() 
		# Exit the loop if the video ends
		if not success:
			break
		# 
		outputs = pre_process(img, net)
		rows = outputs[0].shape[1]
		image_height, image_width = img.shape[:2]

		# Resizing factor for yolov5. 
		x_factor = image_width / INPUT_WIDTH
		y_factor =  image_height / INPUT_HEIGHT

		class_ids = []
		confidences = []
		boxes = []	
		
		
		# Iterate through 25200 detections.
		for r in range(rows):
			row = outputs[0][0][r]
			confidence = row[4]

			# Discard bad detections and continue.
			if confidence >= CONFIDENCE_THRESHOLD:
				classes_scores = row[5:]
				# Get the index of max class score.
				class_id = np.argmax(classes_scores)
	
				#  Continue if the class score is above threshold.
				if (classes_scores[class_id] > SCORE_THRESHOLD):
					confidences.append(confidence)
					class_ids.append(class_id)

					cx, cy, w, h = row[0], row[1], row[2], row[3]

					left = int((cx - w/2) * x_factor)
					top = int((cy - h/2) * y_factor)
					width = int(w * x_factor)
					height = int(h * y_factor)
			     
					box = np.array([left, top, width, height])
					boxes.append(box)
				
		indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
		
		dboxes = []
		dconfs = []
		for i in indices:
			dboxes.append(boxes[i])
			dconfs.append(confidences[i])

		# DeepSORT -> Getting appearence features of the object.
		features = encoder(img, dboxes)
		# DeepSORT -> Storing all the required info in a list.
		detections = [Detection(dbox, dconf, feature) for dbox, dconf, feature in zip(dboxes, dconfs, features)]

		# DeepSORT -> Predicting Tracks. 
		tracker.predict()
		tracker.update(detections)

		# DeepSORT -> Plotting the tracks.
		for track in tracker.tracks:
			if not track.is_confirmed() or track.time_since_update > 1:
				continue
			# DeepSORT -> Changing track bbox to top left, bottom right coordinates
			dbox = list(track.to_tlbr())
			# DeepSORT -> Writing Track bounding box and ID on the frame using OpenCV.
			txt = 'people:' + str(track.track_id)
			
			(label_width, label_height), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
			top_left = tuple(map(int, [int(dbox[0]), int(dbox[1]) - (label_height + baseline)]))
			top_right = tuple(map(int, [int(dbox[0]) + label_width, int(dbox[1])]))
			org = tuple(map(int, [int(dbox[0]), int(dbox[1]) - baseline]))
			
			cv2.rectangle(img, (int(dbox[0]), int(dbox[1])), (int(dbox[2]), int(dbox[3])), (255, 0, 0), 1)
			cv2.rectangle(img, top_left, top_right, (255, 0, 0), -1)
			cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        # Press 'q' to exit the loop
		cv2.imshow("result",img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()