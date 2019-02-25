from darknet import Darknet
import torch
import cv2
from utils import get_all_boxes,nms

class ObjZoneDetectYoloV3:
    def __init__(self,cfgfile,weightfile,gpu_id=0):
        self.__net = Darknet(cfgfile)
        self.__net.eval()
        self.__net.print_network()
        self.__net.load_weights(weightfile)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.__net.cuda(gpu_id)

        self.__shape = (self.__net.width,self.__net.height)
        self.__nms_thresh = 0.1

    def detect(self, im, conf_thresh=0.7):
        im_resized = cv2.resize(im, self.__shape)
        im_rgb = cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB)
        im_torch = torch.from_numpy(im_rgb.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        im_torch = im_torch.to(torch.device("cuda"))
        output = self.__net(im_torch)

        boxes = get_all_boxes(output, self.__shape, conf_thresh, self.__net.num_classes, use_cuda=True)[0]

        boxes = nms(boxes, self.__nms_thresh)

        result = []
        w = im.shape[1]
        h = im.shape[0]
        for i in range(len(boxes)):
            box = boxes[i]

            x1 = int(round((box[0] - box[2] / 2.0) * w))
            y1 = int(round((box[1] - box[3] / 2.0) * h))
            x2 = int(round((box[0] + box[2] / 2.0) * w))
            y2 = int(round((box[1] + box[3] / 2.0) * h))

            x1 = 0 if x1 < 0 else x1
            y1 = 0 if y1 < 0 else y1
            x2 = w - 1 if x2 >= w else x2
            y2 = h - 1 if y2 >= h else y2

            result.append([x1,y1,x2,y2])

        return result

def addRectangle(boxes,im):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r in boxes:
        cv2.rectangle(im,(r[0],r[1]),(r[2],r[3]),(0,0,255),1)

def run():
    cfgfile = "/home/zqp/project/pytorch-0.4-yolov3/cfg/yolo_v3patent.cfg"
    weightfile = "/home/zqp/project/pytorch-0.4-yolov3/cfg/000020.weights"

    detector = ObjZoneDetectYoloV3(cfgfile, weightfile)

    im_path = "/home/zqp/project/pytorch-0.4-yolov3/data-patent/pp6.jpg"
    im = cv2.imread(im_path)

    boxes = detector.detect(im)
    addRectangle(boxes,im)

    cv2.imshow("im",im)
    cv2.waitKey(0)

if __name__=="__main__":
    run()
