set KMP_DUPLICATE_LIB_OK=TRUE


#train

yolo task=detect mode=train model=yolov8m.pt data="C:\Mandy\Scania project\Roboflow\Datasets\DX_uncleangear_step1-2\data.yaml" epochs=300 imgsz=640 patience=300 device=0 pretrained=True workers=0


#Deactivate aug:

yolo task=detect mode=train model=yolov8m.pt data="C:\Mandy\Scania project\Roboflow\Datasets\DX_uncleangear_step1-2\data.yaml" epochs=300 imgsz=640 patience=300 device=0 pretrained=True workers=0 hsv_h=0 hsv_s=0 hsv_v=0 translate=0 scale=0 fliplr=0 mosaic=0 save=True


 yolo task=detect mode=train model=yolov8s.pt data="C:\Mandy\Scania project\Roboflow\Datasets\DX_uncleangear_research-3\data.yaml" epochs=1000 imgsz=640 batch=64 patience=1000 device=0 pretrained=True workers=0 hsv_v=0.3 hsv_s=0.3 degrees=15 scale=0.2  mosaic=0.5 copy_paste=0.3 save=True

copy_paste

Good performance:
 yolo task=detect mode=train model=yolov8m.pt data="C:\Mandy\Scania project\Roboflow\Datasets\DX_uncleangear_research-2\data.yaml" epochs=300 imgsz=640 batch=16 patience=300 device=0 pretrained=True workers=0 degrees=0 save=True

#Good_performance_L:
yolo task=detect mode=train model=yolov8l.pt data="C:\Mandy\Scania project\Roboflow\Datasets\DX_uncleangear_step0-1\data.yaml" epochs=600 imgsz=640 batch=8 patience=600 device=0 pretrained=True workers=0 hsv_v=0.6 hsv_s=0.3 degrees=0 save=True image_weights=True

#L_deactivate_aug:
yolo task=detect mode=train model=yolov8l.pt data="C:\Mandy\Scania project\Roboflow\Datasets\DX_uncleangear_step1-2\data.yaml" epochs=600 imgsz=640 patience=600 device=0 pretrained=True workers=0 hsv_h=0 hsv_s=0 hsv_v=0 translate=0 scale=0 fliplr=0 mosaic=0 save=True batch=8 image_weights=True

try:
yolo task=detect mode=train model=yolov8m.pt data="C:\Mandy\Scania project\Roboflow\Datasets\DX_uncleangear_research-2\data.yaml" epochs=1000 imgsz=640 batch=16 patience=1000 device=0 pretrained=True workers=0 hsv_v=0.5 hsv_s=0.2 degrees=0 save=True



#val
yolo task=detect mode=val model=runs/detect/final_train123v8mpretrained300/weights/best.pt data=../Datasets/DX_uncleangear_research-2/data.yaml device=0 


#Deburring

#train
yolo task=detect mode=train model=yolov8l.pt data="C:\Mandy\Scania project\Roboflow\Datasets\Deburring\DX_deburring_step12-v2\data.yaml" epochs=600 imgsz=640 batch=8 patience=600 device=0 pretrained=True workers=0 hsv_v=0.6 hsv_s=0.3 degrees=0 save=True image_weights=True

yolo task=detect mode=train model=yolov8l.pt data="C:\Mandy\Scania project\Roboflow\Datasets\Deburring\DX_deburring_step12-v2\data.yaml" epochs=600 imgsz=640 patience=600 device=0 pretrained=True workers=0 hsv_h=0 hsv_s=0 hsv_v=0 translate=0 scale=0 fliplr=0 mosaic=0 save=True batch=8 image_weights=True


#val
yolo task=detect mode=val model=./runs/detect/train123_l/weights/best.pt data=../Datasets/Deburring/DX_deburring_step12-v2/data.yaml device=0 workers=0

#predict
yolo task=detect mode=predict model=./runs/detect/train123_l/weights/best.pt conf=0.5 source=../Datasets/Deburring/DX_deburring_step12-v2/valid/images workers=0 save_conf=True