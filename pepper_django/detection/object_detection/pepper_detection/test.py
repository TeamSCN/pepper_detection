import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

#MODEL_NAME = '/home/cc/pepper/django/pepper/detection/object_detection/pepper_detection/ssd_mobilenet_v1_coco_11_06_2017'
MODEL_NAME=os.path.dirname(os.path.abspath(__file__))+'/ssd_mobilenet_v1_coco_11_06_2017/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.dirname(os.path.abspath(__file__))+'/data/mscoco_label_map.pbtxt' #os.path.join('data', 'mscoco_label_map.pbtxt')
print(PATH_TO_LABELS)
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("../..")

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


import sys
sys.path.append('/home/cc/pepper/pynaoqi-python2.7-2.5.5.5-linux64/lib/python2.7/site-packages')# import naoqi package URL

from naoqi import ALProxy
import numpy as np
from matplotlib import pyplot as plt



NUM_CLASSES = 90

IMAGE_SIZE = (12, 8)



label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def detect_object(image_np): #detect picture basic function
    #sess=tf.Session(graph=detection_graph)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    with sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        boxes=np.squeeze(boxes)
        classes=np.squeeze(classes).astype(np.int32)
        scores=np.squeeze(scores)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

    return image_np,boxes,classes,scores #return image and detect object class and score

def capture_pic(): #capture a picure from pepper
    IP = "192.168.1.140"
    PORT = 9559
    AL_kTopCamera = 0
    AL_kQVGA = 2  # picture size 640x480
    AL_kBGRColorSpace = 13
    videoDevice = ALProxy('ALVideoDevice', "192.168.1.140", 9559)

    # subscribe top camera
    nameId='test3'
    captureDevice = videoDevice.subscribeCamera(
        nameId, AL_kTopCamera, AL_kQVGA, AL_kBGRColorSpace, 20)

    # create image
    width = 640
    height = 480
    image = np.zeros((height, width, 3), np.uint8)
    result = videoDevice.getImageRemote(captureDevice);
    if result == None:
        print 'cannot capture.'
    elif result[6] == None:
        print 'no image data string.'

    values = map(ord, list(result[6]))
    i = 0
    for y in range(0, height):
        for x in range(0, width):
            image.itemset((y, x, 0), values[i + 0])
            image.itemset((y, x, 1), values[i + 1])
            image.itemset((y, x, 2), values[i + 2])
            i += 3
    videoDevice.releaseImage(captureDevice)
    videoDevice.unsubscribe(captureDevice)
    return image

'''if __name__ == '__main__':


    image=Image.open('test_images/image10.jpg')
    #image_np=capture_pic()
    image_np=load_image_into_numpy_array(image)
    result,boxes,classes,scores=detect_object(image_np)
    print(scores)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(result)
    plt.show()'''