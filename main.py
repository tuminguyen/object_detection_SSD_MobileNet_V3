from datetime import datetime, date
import cv2
from config import *
import argparse
import json
import numpy as np

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
                        type=float, help='confident threshold')
    parser.add_argument('--incl', '-incl', nargs='+',
                        help='list of interested objects, all lowercase')
    parser.add_argument('--dump', '-d', type=bool, default=False,
                        help='dump log to file or not. True | False')
    parser.add_argument('--rate', '-r', type=int, default=1,
                        help='Frame rate to process')
    args = parser.parse_args()
    return vars(args)


def my_converter(obj):
    '''
    fix ERROR: Object of type X is not JSON serializable when dumping json
    :param obj:
    :return:
    '''
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.__str__()
    elif isinstance(obj, date):
        return obj.__str__()


def detect(img, interested=[], thresh=0.6):
    classes, confidences, boxes = net.detect(img, confThreshold=thresh, nmsThreshold=0.05)
    if interested is None:
        interested = objects
    if len(classes):
        for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
            label = objects[classId - 1]
            if label in interested:
                if arguments['dump']:
                    doc = {"date": date.today(),
                           "time": datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S'),
                           "object": label,
                           "count": len(classes),
                           "confidence": confidence}
                    return doc
                cv2.rectangle(img, box, color=(0, 255, 255))
                cv2.putText(img, label, (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 175, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 255, 0), 2)

                return img


if __name__ == '__main__':
    cap = cv2.VideoCapture('/dev/video0')  # video2: usb cam / video0: laptop cam or pi cam
    cap.set(3, 640)
    cap.set(4, 480)
    arguments = init_parse()
    fps = cap.get(cv2.CAP_PROP_FPS)  # fps = 30
    frameCount = 0
    if arguments['img'] is None or arguments['img'] == '':
        with open("log.json", 'a+') as f_out:
            while True:
                frameCount += 1
                ret, frame = cap.read()
                if not ret:
                    break
                if frameCount % arguments['rate'] == 0:  # detect on every [rate] frames => 30/rate times per second
                    detected = detect(img=frame, interested=arguments['incl'], thresh=arguments['thresh'])
                    print(detected)
                    if arguments['dump']:
                        if detected is not None:
                            json.dump(detected, f_out, default=my_converter)
                            f_out.write("\n")
                    else:
                        if detected is not None:
                            cv2.imshow('out', detected)
                        else:
                            cv2.imshow('out', frame)
                        k = cv2.waitKey(33)
                        if k == 27:  # press ESC
                            break
    else:
        frame = cv2.imread(arguments['img'])
        detected_img = detect(img=frame, interested=arguments['incl'], thresh=arguments['thresh'])
        cv2.imshow('out', detected_img)
        cv2.waitKey(1000)
