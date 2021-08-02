import cv2
import numpy as np


class IdCardDetection():
    def __init__(self, whT=416, confThreshold=0.5, nmsThreshold=0.3,modelConfiguration='',modelWeights='',classesFile=''):

        self.whT = whT
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.modelConfiguration = modelConfiguration
        self.modelWeights = modelWeights
        self.classesFile = classesFile







    def SetNetConf(self, img):


        # print(self.classNames)
        # print(len(self.classNames))

        net = cv2.dnn.readNetFromDarknet(self.modelConfiguration, self.modelWeights)
        # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (self.whT, self.whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)

        layerNames = net.getLayerNames()
        # print(layerNames)
        outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        # print(net.getUnconnectedOutLayers())
        # print(outputNames)
        outputs = net.forward(outputNames)

        return outputs

    def findObjects(self, outputs, img):
        hT, wT, cT = img.shape
        bbox = []
        classIds = []
        confs = []
        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    classIds.append(classId)
                    confs.append(float(confidence))

        # print(len(bbox))
        indices = cv2.dnn.NMSBoxes(bbox, confs, self.confThreshold, self.nmsThreshold)

        # print(indices)

        return indices, confs, classIds, bbox


def main():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    detector = IdCardDetection(modelConfiguration='yolo/yolov3_training.cfg',
                               modelWeights='yolo/yolov3_training_last_1200.weights', classesFile='yolo/coco_2.names')

    classNames = []
    with open('yolo/coco_2.names', 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    while True:
        success, img = cap.read()
        outputs = detector.SetNetConf(img)
        indices, confs, classIds, bbox = detector.findObjects(outputs, img)
        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.imshow("img", img)
        cv2.waitKey(1)






if __name__ == "__main__":
    main()



