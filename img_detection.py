import cv2
import matplotlib.pyplot as plt

config_file = 'ssd/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classlabels = []
file_name = 'Labels.txt'
with open(file_name, 'rt') as fpt:
    classlabels = fpt.read().rstrip('\n').split('\n')
    
model.setInputSize(320,320)
model.setInputScale(1.0/127.5) ##255/2=127.5
model.setInputMean((127.5,127.5,127.5)) ## mobilenet => [-1,1]
model.setInputSwapRB(True) ## convert auto to RGB

img = cv2.imread('photos/car_dog_person.jpg')
ClassIndex, confidece, bbox = model.detect(img, confThreshold=0.5)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
    #cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 0), 2)
    #putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)
    cv2.rectangle(img,boxes,(255,0, 0), 2) 
    cv2.putText(img,classlabels[ClassInd-1],(boxes[0]+10,boxes[1]+40), font, fontScale=font_scale,color=(0, 255,0), thickness=3)

# show the output image
cv2.imshow("Output", img)
cv2.waitKey(0)
