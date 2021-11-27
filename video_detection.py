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

cap = cv2.VideoCapture("video/london-westminsterbridge3.mp4")

# Check if the video is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open video")
    
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret,frame = cap.read()
    ClassIndex, confidece, bbox = model.detect(frame,confThreshold=0.55)
    print(ClassIndex)
    if(len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            if (ClassInd<=80):
                cv2.rectangle(frame,boxes,(255,0, 0), 2)
                cv2.putText(frame,classlabels[ClassInd-1],(boxes[0]+10,boxes[1]+40), font,
                 fontScale=font_scale,color=(0, 255,0), thickness=3)
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllwindows()
