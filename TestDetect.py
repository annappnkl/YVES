import cv2, os
import numpy as np
import tensorflow as tf
from Test import detect_fn, files
from object_detection.utils import label_map_util

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

while cap.isOpened(): 
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1

    scores = detections['detection_scores']
    detection = detections['detection_classes']

        
    if len(detection) > 0:
        score = scores[0]
        labelobj =  category_index[detection[0] + label_id_offset] 
        label =labelobj['name']
        
        boxes = detections['detection_boxes']
        box = boxes[0]
        y1 = box[0] * 620 # in prozent vom gesamtbild?
        x1 = box[1] * 620
        y2 = box[2] * 620
        x2 = box[3] * 620
        
        print(label, score, y1, x1, y2, x2)

    else:
        print("no detection")