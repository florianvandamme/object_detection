import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import cv2
import time

# Subdirectory imports
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Check whether an argument was passed. If none was passed use the
# default model. First time you will still need to download the thing manually
# just run "> python detect.py ssd_mobilenet_v1_coco_11_06_2017"
try:
    sys.argv[1]
except IndexError:
    MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
else:
    MODEL_NAME = sys.argv[1]
    print(MODEL_NAME)
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# Load model in TF memory
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Fancy parse on the labels
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Start capturing webcam input with OpenCV
cap = cv2.VideoCapture(0)
now = time.time()
# Empty arrays for the first draw
boxes = []
classes =[]
scores = []

# User the model
with detection_graph.as_default():
    # To create a TensorFlow session
    with tf.Session(graph=detection_graph) as sess:
        # Run forever and ever and ever and ever and ever
        while(True):
            # Fetch the input for the OpenCV Cam stream, this is already array-able material
            ret, image_np = cap.read()
            # Only re-calculate every .5 secs, easy performance hack
            if now + .5 < time.time():
                # Reset current time when
                now = time.time()
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            # Ouput the frame with the detection box drawn on to the CV2 canvas
            cv2.imshow('Object Detection', image_np)

            # Listen for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Clean delete if done or exit
cap.release()
cv2.destroyAllWindows()








