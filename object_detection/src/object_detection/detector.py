import rospy
import torch
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from rospy.exceptions import ROSInitException 

class Detector:
    def __init__(self, config):
        self.architecture   = config["architecture"]
        self.model          = config["model"]
        self.checkpoint     = config["checkpoint"]
        self.device         = config["device"]
        self.confident      = config["confident"]
        self.iou            = config["iou"]
        self.classes        = config["classes"]
        self.detector       = None

        if self.architecture == 'yolo':
            self.detector = torch.hub.load('ultralytics/yolov5', self.model , device=self.device) # 'yolov5n'
            self.detector.conf = self.confident
            self.detector.iou = self.iou
            self.detector.classes = self.classes
        elif self.architecture == 'detectron':
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file(self.model)) # "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            self.checkpoint  = self.model
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.checkpoint)
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.iou 
            self.cfg.MODEL.DEVICE = self.device
            self.detector = DefaultPredictor(self.cfg)

            # raise rospy.exceptions.ROSInitException("Detectron architecture was not implemented yet.")
        else:
            raise rospy.exceptions.ROSInitException("Unrecognised architecture.")

    def detect(self, image):
        if self.architecture == 'yolo':
            return self.detector(image).render()[0]
        elif self.architecture == 'detectron':
            outputs = self.detector(image)
            v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            return out.get_image()[:, :, ::-1]