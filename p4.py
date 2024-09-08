# from ultralytics import YOLO
# import cv2
# import cvzone
# import math

# # model=YOLO('yolov8l.pt')
# # results=model("dfhb.jpg",show=True)
# # cv2.waitKey(0)

# cap=cv2.VideoCapture("ppe-1-1.mp4")
# cap.set(3,1280)
# cap.set(4,720)

# model= YOLO("yolov9e.pt")
# while True:
#     _,frame=cap.read()
#     results=model(frame,stream=True)
#     for r in results:
#         boxes=r.boxes
#         for box in boxes:
#             x1,y1, x2,y2=box.xyxy[0]
#             x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            
#             # Using cv2
#             # cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),3)
            
#             # Using cvzone
#             w,h=x2-x1,y2-y1
#             cvzone.cornerRect(frame,(x1,y1,w,h))
            
#             conf=math.ceil(box.conf[0]*100)/100
#             print(conf)
            
#     cv2.imshow("img",frame)
#     cv2.waitKey(1)






from ultralytics import YOLO
import cv2
import cvzone
import math

# import time

model=YOLO('best (1).pt')
results=model("construction-worker-at-india-2G9EB2A.jpg",show=True)


# annotated_image = results[0].plot()

# # Save the annotated image
# cv2.imwrite("output_image.jpg", annotated_image)

# # Display the annotated image (optional)
# cv2.imshow("Annotated Image", annotated_image)


cv2.waitKey(0)

# cap = cv2.VideoCapture("ppe-2-1.mp4")  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
# # cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video
 
# ad66718bdf57baea6c47817edb0e90b7
# osho
# depositphotos_477967708-stock-photo-mumbai-maharashtra-india-december-2019
# xoPt4lufgldCsgDbbzSDQgz00qsjxrsl7pSRnjKnXEg.webp
# main-qimg-b34abe3e13e17fc8e192a53aa0a88663-lq.jpeg
# 65f2c58465abf032ae52a4fb8b591ad2.jpeg
# construction-worker-at-india-2G9EAX8.jpg
# construction-worker-at-india-2G9EB2A.jpg
# depositphotos_477968842-stock-photo-mumbai-maharashtra-india-december-2019
# 306ba3c759e3a61cf229935421e2241b
# a8bbc6ab0b48a9e2909bcdbc2aa4c2d1
# f54184cdd8b6db9ea947b58a3804a214.jpg
# daily-wage-construction-worker-working-footage-154644524_prevstill
# 4993b3c65cbbacb7017336b63a5973f4
# 55ffe77e2b45df15898fbf28cc26bffa
# 1682871ba354a5bd500ef009a2b260d1
# f80605af10c71ebb5ad4cbf89e80713b
# d3f2482eaacec8890980af817242e2e6
# 102fca5d803fc9bb57c6840f1c04c07d
# 820c99b339620072491d6e2c8b281cad
# 5f4db600eddd3eb0341a1d2d56fae97a
# 741f38eaacf10c0f21b60d55dc81555f
# manual-workers-doing-laborious-job-at-a-rural-construction-site-in-new-delhiindia-2GGMCM6
# istockphoto-610442626-612x612
# indian-village-man-worker-holded-bricks-home-construction-site-india-dec-168123791
# eb00d2a0cbdc0373308b76852df836d9




# model = YOLO("best (1).pt")
# classNames = ['Hardhat', 'NO-Hardhat']
 
# prev_frame_time = 0
# new_frame_time = 0
 
# while True:
#     # new_frame_time = time.time()
#     _, frame = cap.read()
#     results = model(frame, stream=True)
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             # Bounding Box
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
#             w, h = x2 - x1, y2 - y1
#             cvzone.cornerRect(frame, (x1, y1, w, h))
#             # Confidence
#             conf = math.ceil((box.conf[0] * 100)) / 100
#             # Class Name
#             cls = int(box.cls[0])
 
#             cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
 
#     # fps = 1 / (new_frame_time - prev_frame_time)
#     # prev_frame_time = new_frame_time
#     # print(fps)
 
#     cv2.imshow("Image", frame)
#     cv2.waitKey(1)
#     k = cv2.waitKey(60) & 0xff
#     if k == ord('q'):
#         break


# import torch
# import torchvision
# from torchvision.models.detection import fasterrcnn_resnet50_fpn
# import cv2
# import cvzone
# import math

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# # Load the pre-trained Faster R-CNN model
# model = fasterrcnn_resnet50_fpn(pretrained=True)
# model.eval()
# model.to(device)

# # Define the class labels for your dataset
# classNames = ['__background__', 'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
#               'Safety Vest', 'machinery', 'vehicle']

# cap = cv2.VideoCapture("ppe-1-1.mp4")  # Specify the path to your video file
# cap.set(3, 1280)
# cap.set(4, 720)

# while True:
#     _, frame = cap.read()

#     # Preprocess the frame
#     input_img = torchvision.transforms.ToTensor()(frame).to(device)
#     input_img = input_img.unsqueeze(0)  # Add a batch dimension

#     # Perform inference
#     with torch.no_grad():
#         predictions = model(input_img)

#     # Process the predictions
#     for idx, pred in enumerate(predictions[0]['boxes']):
#         x1, y1, x2, y2 = map(int, pred.cpu().numpy())
#         w, h = x2 - x1, y2 - y1
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
#         cvzone.cornerRect(frame, (x1, y1, w, h))
#         cls_idx = int(predictions[0]['labels'][idx])
#         conf = float(predictions[0]['scores'][idx])
#         print("Class index:", cls_idx)
#         if cls_idx < len(classNames):
#             cls_name = classNames[cls_idx]  # Corrected the indexing
#             cvzone.putTextRect(frame, f'{cls_name} {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
#         else:
#             print("Invalid class index:", cls_idx)


#     cv2.imshow("Image", frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
