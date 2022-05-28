import cv2
import time
import numpy as np
import os

classes = ['adhaar_card','pan_card', 'face']
confidenceThresh = 0.5
nmsthreshold = 0.4
base_dir = os.path.dirname(__file__)

def get_output_layers(yolo_config_file, yolo_weights):
    net = cv2.dnn.readNetFromDarknet(yolo_config_file, yolo_weights)

    if net == None:
        print("Error in retrieving the weights. Please check your YOLO config files")

    # output layer names from YOLO darknet
    layers = net.getLayerNames()

    print('ln----')
    ln = [layers[i - 1] for i in net.getUnconnectedOutLayers()]
    return ln, net

def get_detection_layer(yolo_config_file, yolo_weights):
    output_layers, net = get_output_layers(yolo_config_file, yolo_weights)

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (608, 608), (0,0,0),
        swapRB=True, crop=False)
    net.setInput(blob)

    start = time.time()
    layerOutputs = net.forward(output_layers)
    end = time.time()

    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    return layerOutputs

def get_bounding_boxes(layerOutputs):
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
        # extract the class ID and confidence (i.e., probability) of
        # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            confidence = scores[classID]
            # print("classID", classID, confidence)  

            #Checking the confidence. Could be better if trained on more images of PAN and adhaar.
            if confidence > 0.5:
                # detection from YOLO has center co-ordinates and width, height of the box
                
                print("prediction----", classes[classID], confidence)
                print(scores)
                # scale the bounding box coordinates back relative to the size of the image
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))  #left corner co-ordinate
                y = int(centerY - (height / 2)) #top corner co-ordinate

                # print('xy', x, y)
                # update list of bounding box coordinates, confidences and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    return boxes, confidences, classIDs

def predict_image(boxes, confidences, classIDs):
    # Remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThresh, nmsthreshold)

    for idx in indices:
        print(idx)
        box = boxes[idx]
        left, top, width, height = box[0], box[1], box[2], box[3]
        right, bottom = top + height, left + width
        print('left top----', left, top)

        extract_prediction_images(idx, left, top, right, bottom)  #extract the documents detected in the image
        drawPredictionBox(classIDs[idx], confidences[idx], left, top, right, bottom)

def drawPredictionBox(class_id, confidence, left, top, right, bottom):
    label = str(classes[class_id])
    confidence = round(confidence, 2)
    color = (0,0,255)

    cv2.rectangle(img, (left,top), (bottom,right), color, 2)

    cv2.waitKey(0)
    cv2.putText(img, label+" "+str(confidence), (left+10,bottom+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def extract_prediction_images(index, left, top, right, bottom):
    filename = img_path.split('\\')[-1]

    if left<0 or top<0:
        left, top = 0, 0
    if right<0 or bottom<0:
        right, bottom = 0,0

    crop_image = image_copy[top:right, left:bottom]

    cv2.imwrite(base_dir+'/updated_images/'+ str(index) + "-" + filename, crop_image)
    # cv2.imshow('image', crop_image)

if __name__ == '__main__':

    img_path = r"C:\Users\utk09\OneDrive\Desktop\object_extractor\example_test_images\WhatsApp Image 2022-05-09 at 12.39.02 AM.jpeg"
    img = cv2.imread(img_path)
    print(img_path.split('\\')[-1])
    img = cv2.resize(img,(608,608),  interpolation=cv2.INTER_LINEAR)

    print(img.shape)
    image_copy = img.copy()
    (H, W) = img.shape[:2]

    print('aspect ratio-------', W/H)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)

    yolo_config_file = "C:/Users/utk09/OneDrive/Desktop/object_extractor/darknet/cfg/yolov3_custom_new.cfg"
    yolo_weights = "C:/Users/utk09/OneDrive/Desktop/object_extractor/backup/yolov3_custom_new_last.weights"
    layerOutput = get_detection_layer(yolo_config_file, yolo_weights)

    boxes, confidences, classIDs = get_bounding_boxes(layerOutput)

    predict_image(boxes, confidences, classIDs)
    #displaying the resultant image
    cv2.imshow('bounding box prediction', img)
    cv2.waitKey(0)

