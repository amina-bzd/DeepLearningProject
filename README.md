# DeepLearningProject
Detection and classification of mobile objects by deep learning

Object detection is a computer technology related to computer vision and image processing that deals with the detection of instances of semantic objects of a certain class (such as humans, buildings or cars) in digital images and videos.

In this project we will detect and classify objects in an image, video and a live webcam. Using OpenCV's Deep Neural Network (dnn) module to load frozen Tenserflow models (pre-trained deep learning architecture) using an SSD-MobileNetv3 model trained on a COCO dataset capable of detecting objects of 80 common classes.

To detect moving objects we need:

  1) SSD-MobileNetv3 model: dep learning architecture model based on TensorFlow.
  2) Existing configuration file for the model: OpenCV needs an additional configuration file to import object detection models from TensorFlow. It is based on a text version of the same graph serialized in protocol buffers (protobuf) format.
  3) Coco.names labels: The names of the objects of 80 classes belong to the COCO Dataset.
  
