import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, strip_optimizer
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel

source = "bob1.mp4"
save_dir = "C:/Users/tadas/Desktop/YOLOv7/yolov7-main/bob11.mp4"



weights = "yolov7.pt"
conf_thres = 0.25
iou_thres = 0.45
classes = range(80)
agnostic_nms = True
augment = True
img_size = 640
imgsz = 640
trace = True
update = True

def detect():

    
    # Initialize
    #set_logging()
    device = select_device("0")
    half = device.type != 'cpu'  # half precision only supported on CUDA
    imgsz = img_size

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, img_size)

    if half:
        model.half()  # to FP16

    
    video = cv2.VideoCapture(source) 
    frame_width = int(video.get(3)) 
    frame_height = int(video.get(4))
    size = (frame_width, frame_height) 
    forc = cv2.VideoWriter_fourcc(*'mp4v')
    result = cv2.VideoWriter(save_dir,  forc, 10, (size)) 

    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    colors = (0, 200, 100)

    #Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        imgg = img
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)


        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()


        # Process detections
        for i, det in enumerate(pred):  # detections per image

            #p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)   
            p, s, im0 = path, '', im0s   
        
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    #bounding box cordinates
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    #bonding box center
                    center = ( int((c1[0] + c2[0]) / 2), int((c1[1] + c2[1]) / 2 ))
                    #print(names[int(cls)])
                    if names[int(cls)] == "boat":
                        radius = 5
                        color = (255, 0, 0)
                        thickness = 2

                        im0 = cv2.circle(im0, center, radius, color, thickness)
                        #if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        plot_one_box(xyxy, im0, label=label, color=colors, line_thickness=1)

            # Print time (inference + NMS)
            #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            #while cv2.waitKey(33) != ord('q'):
            result.write(im0)
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond
    

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    with torch.no_grad():
        if update:  # update all models (to fix SourceChangeWarning)
            for weights in ['yolov7.pt']:
                detect()
                strip_optimizer(weights)
        else:
            detect()

