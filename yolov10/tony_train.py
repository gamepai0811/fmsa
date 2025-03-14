from ultralytics import YOLOv10


# If you want to finetune the model with pretrained weights, you could load the 
# pretrained weights like below
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')


#Ours (YOLOv10n + FMSA)
#model = YOLOv10('ultralytics/cfg/models/v10/yolov10exV5_tf.yaml')
#model.train(data='ultralytics/cfg/datasets/VisDrone.yaml', epochs=1000, batch=4, imgsz=640, device='0,1', optimizer='SGD', lr0=0.01, momentum=0.937, pretrained=False)

#Ours (YOLOv10n + FMSA + FUS)
#model = YOLOv10('ultralytics/cfg/models/v10/yolov10exV5_up.yaml')
#model.train(data='ultralytics/cfg/datasets/VisDrone.yaml', epochs=1000, batch=4, imgsz=640, device='0,1', optimizer='SGD', lr0=0.01, momentum=0.937, pretrained=False)

#Ours (YOLOv10n + FMSA + FDS)
#model = YOLOv10('ultralytics/cfg/models/v10/yolov10exV5_down.yaml')
#model.train(data='ultralytics/cfg/datasets/VisDrone.yaml', epochs=1000, batch=4, imgsz=640, device='0,1', optimizer='SGD', lr0=0.01, momentum=0.937, pretrained=False)

#Ours (YOLOv10n + FMSA + FUS + FDS)
model = YOLOv10('ultralytics/cfg/models/v10/yolov10exV5.yaml')
model.train(data='ultralytics/cfg/datasets/VisDrone.yaml', epochs=1000, batch=4, imgsz=640, device='0,1', optimizer='SGD', lr0=0.01, momentum=0.937, pretrained=False)



