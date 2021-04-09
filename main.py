import cv2
from config import *
import argparse

objects = list()
with open(COCO_OBJECTS_FILE, 'r') as f:
    objects = f.read().rstrip('\n').split('\n')

net = cv2.dnn_DetectionModel(WEIGHT_PATH, CONFIG_PATH)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def init_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-i',
                        type=str, help='image path')
    parser.add_argument('--thresh', '-t', default=0.6,
                        type=float, help='image path')
    parser.add_argument('--incl', '-incl', nargs='+',
                        help='list of interested objects, all lowercase')
    args = parser.parse_args()
    return vars(args)


def detect(img, interested=[], thresh=0.6):
    classes, confidences, boxes = net.detect(img, confThreshold=thresh, nmsThreshold=0.05)
    if interested is None:
        interested = objects
    if len(classes):
        for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
            # print(objects[classId - 1], confidence)
            label = objects[classId - 1]
            if label in interested:
                cv2.rectangle(img, box, color=(0, 255, 255))
                cv2.putText(img, label, (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 175, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 255, 0), 2)
    return img


if __name__ == '__main__':
    cap = cv2.VideoCapture('/dev/video2')  # video2: usb cam / video0: laptop cam
    cap.set(3, 640)
    cap.set(4, 480)
    arguments = init_parse()
    if arguments['img'] is None or arguments['img'] == '':
        while True:
            ret, frame = cap.read()
            print(cap.get(cv2.CAP_PROP_FPS))
            if not ret:
                break
            detected_img = detect(img=frame, interested=arguments['incl'], thresh=arguments['thresh'])
            cv2.imshow('out', detected_img)
            k = cv2.waitKey(33)
            if k == 27:  # press ESC
                break
    else:
        frame = cv2.imread(arguments['img'])
        detected_img = detect(img=frame, interested=arguments['incl'], thresh=arguments['thresh'])
        cv2.imshow('out', detected_img)
        cv2.waitKey(1000)
