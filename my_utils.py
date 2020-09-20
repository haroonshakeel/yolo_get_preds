import cv2
import imutils
from imutils.video import FPS
import numpy as np

def get_yolo_preds(net, input_vid_path="input_video/japan.mp4", output_vid_path="output_video/yolo_output.avi", confidence_threshold=0.5, overlapping_threshold=0.3, write_output=False, show_display=True, labels = None):
    # Get layer names that output predictions from YOLO
    # List of colors to represent each class label with distinct color
    np.random.seed(123)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    W = None
    H = None
    cap = cv2.VideoCapture(input_vid_path)

    if (cap.isOpened() == False):
        print("Error opening video stream or file")
        return

    (success, frame) = cap.read()
    # frame = imutils.resize(frame, width=640)
    if write_output:

        out = cv2.VideoWriter(output_vid_path, cv2.VideoWriter_fourcc(
            *"MJPG"), cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]), True)

    fps = FPS().start()

    while success:
        # frame = imutils.resize(frame, width=640)
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # Construct blob of frames by standardization, resizing, and swapping Red and Blue channels (RBG to RGB)
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > confidence_threshold:
                    # Scale the bboxes back to the original image size
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # Remove overlapping bounding boxes and boundig boxes
        bboxes = cv2.dnn.NMSBoxes(
            boxes, confidences, confidence_threshold, overlapping_threshold)
        if len(bboxes) > 0:
            for i in bboxes.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if show_display:
            cv2.imshow("Predictions", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break the loop
            if key == ord("q"):
                break

        if write_output:
            out.write(frame)

        fps.update()
        (success, frame) = cap.read()
    fps.stop()
    print("Elasped time: {:.2f}".format(fps.elapsed()))
    print("FPS: {:.2f}".format(fps.fps()))
    cap.release()
    if write_output:
        out.release()
    cv2.destroyAllWindows()