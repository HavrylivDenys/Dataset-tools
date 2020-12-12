import cv2
import sys
import os
from tensorflow.keras.models import Model, load_model
import tensorflow.keras
import numpy as np
from math import atan2, degrees


def extract_boxes_confidences_classids(outputs, confidence, width, height):
    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        for detection in output:
            # Extract the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]

            # Consider only the predictions that are above the confidence threshold
            if conf > confidence:
                # Scale the bounding box back to the size of the image
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')

                # Use the center coordinates, width and height to get the coordinates of the top left corner
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                classIDs.append(classID)

    return boxes, confidences, classIDs


def draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors, labels, small=False):
    if len(idxs) > 0:
        for i in idxs.flatten():
            color = [int(c) for c in colors[classIDs[i]]]

            if small:
                w, h = 2, 2
                x, y = int(boxes[i][0] + boxes[i][2] / 2), int(boxes[i][1] + boxes[i][3] / 2)
            else:
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]

            # draw the bounding box and label on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    return image


def make_prediction(net, layer_names, labels, image, confidence, threshold):
    height, width = image.shape[:2]

    # Create a blob and pass it through the model
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (608, 608), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    # Extract bounding boxes, confidences and classIDs
    boxes, confidences, classIDs = extract_boxes_confidences_classids(outputs, confidence, width, height)

    # Apply Non-Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    return boxes, confidences, classIDs, idxs


def cut_image(img, boxes, idxs):
    images = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract bounding box coordinates (x, y) - top left
            x, y = boxes[i][0] - int(boxes[i][2] * 0.6), boxes[i][1] - int(boxes[i][3] * 0.6)
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            w, h = int(boxes[i][2] * 2.4), int(boxes[i][3] * 2.4)
            new_img = img[y: y + h, x:x + w]

            images.append((new_img, (x, y, w, h), i))

        return images
    else:
        return img.copy(), (0, 0, img.shape[0], img.shape[1])


def extractImages(src):
    count = 0
    capture = cv2.VideoCapture(src)
    success, image = capture.read()
    video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')

    capture.set(cv2.CAP_PROP_POS_MSEC, (count * 1000 / 24))  # added this line
    fps = capture.get(5)

    out = cv2.VideoWriter('project.avi', fourcc, fps, (int(capture.get(3)), int(capture.get(4))))

    net = cv2.dnn.readNetFromDarknet("models/test/bb.cfg", "models/test/bb.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    net2 = cv2.dnn.readNetFromDarknet('models/test/cats.cfg', 'models/test/cats.weights')
    net2.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    net3 = cv2.dnn.readNetFromDarknet('models/test/dogs.cfg', 'models/test/dogs.weights')
    net3.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net3.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    layer_names2 = net2.getLayerNames()
    layer_names2 = [layer_names2[i[0] - 1] for i in net2.getUnconnectedOutLayers()]

    labels = ["cat", "dog"]

    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    while capture.isOpened():
        if count > (video_length - 1):
            break
        success, image = capture.read()
        boxes, confidences, classIDs, idxs_m = make_prediction(net, layer_names, labels, image, 0.3,
                                                               0.5)
        image = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs_m, colors, labels)
        images = cut_image(image, boxes, idxs_m)

        labels2 = ["p0", "p1", "p2", "p3", "p4"]
        colors = np.random.randint(0, 255, size=(len(labels2), 3), dtype='uint8')
        id = -1

        if len(idxs_m) != 0:
            for obj in images:
                img = obj[0]
                size = obj[1]
                if id == -1:
                    id = obj[2]

                if id == 1:
                    boxes, confidences, classIDs, idxs = make_prediction(net2, layer_names2, labels2, img, 0.1,
                                                                         0.7)
                    img = draw_bounding_boxes(img, boxes, confidences, classIDs, idxs, colors, labels2, small=True)
                else:
                    boxes, confidences, classIDs, idxs = make_prediction(net3, layer_names2, labels2, img, 0.1,
                                                                         0.7)
                    img = draw_bounding_boxes(img, boxes, confidences, classIDs, idxs, colors, labels2, small=True)

                image[size[1]: size[1] + size[3], size[0]: size[0] + size[2]] = img

        print('Read a new frame: ', success, count)
        count = count + 1
        out.write(image)

    out.release()


path = "samples_video"
file_name = "1.mp4"
full_path = os.path.join(path, file_name)

extractImages(full_path)
