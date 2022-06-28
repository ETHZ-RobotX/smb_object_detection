import torch
import numpy as np

# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog

from rospy.exceptions import ROSInitException 
from yolov5.models.common import DetectMultiBackend, AutoShape
from yolov5.utils.torch_utils import select_device

class ObjectDetector:
    def __init__(self, config):

        self.architecture       = config["architecture"]
        self.model              = config["model"]
        self.model_path         = config["model_path"]
        self.checkpoint         = config["checkpoint"]
        self.device             = config["device"]
        self.confident          = config["confident"]
        self.iou                = config["iou"]
        self.classes            = config["classes"]
        self.multiple_instance  = config["multiple_instance"]
        self.detector           = None

        if self.architecture == 'yolo':
            if self.model_path:
                print("Path: ", self.model_path + self.model + ".pt")
                print("Device: ", self.device)
                device = select_device(self.device)
                self.detector = DetectMultiBackend(self.model_path + "/" + self.model + ".pt", device=device)
                self.detector = AutoShape(self.detector)
                self.detector = self.detector.to(device)
            else:
                print("No model path defined, loading from hub")
                self.detector = torch.hub.load('ultralytics/yolov5', self.model , device=self.device) # 'yolov5n'
            self.detector.conf = self.confident
            self.detector.iou = self.iou
            self.detector.classes = self.classes

        elif self.architecture == 'detectron':

            raise ROSInitException("Detectron return type was not adapted. \
                                                     See 'detect' function. \
                                                     You are welcomed to implement :)")

            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file(self.model)) # "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            self.checkpoint  = self.model
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.checkpoint)
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.iou 
            self.cfg.MODEL.DEVICE = self.device
            self.detector = DefaultPredictor(self.cfg)

        else:
            raise ROSInitException("Unrecognised architecture.")
    
    def filter_detection(self, detection):
        """ Detects objects on the given image using set model (YOLO V5 or Detectron) 
        Args:
            detection: object infos in Pandas data frame

        Returns:
            detection: one instance with the highest confidance for every class.

        """
        # pick the one with highest confidance for every class.
        detected_objects = []
        row_to_delete    = []
        for i in range(len(detection)):
            if detection['class'][i] in detected_objects:
                row_to_delete.append(i)
            else:
                detected_objects.append(detection['class'][i])
        
        detection = detection.drop(row_to_delete, axis=0)
        detection.reset_index(inplace=True)

        return detection

    def detect(self, image):
        """ Detects objects on the given image using set model (YOLO V5 or Detectron) 
        Args:
            image             : numpy matrix, RGB or Gray Scale image
            multiple_instance : bool to decide to allow multiple instances for classes.
        
        Returns:
            detection       : object infos in Pandas data frame
            detection image : image with bounding boxes

        """

        if self.architecture == 'yolo':
            output = self.detector(image)
            detection = output.pandas().xyxy[0]

            if not self.multiple_instance:
                detection = self.filter_detection(detection)
            return detection, output.render()[0]

        # elif self.architecture == 'detectron':
        #     outputs = self.detector(image)
        #     v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        #     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #     
        #     # TODO: implement a function to transform output to the panda frame
        #     # output = self.transform2pandaframe(output) 
        #     return outputs, out.get_image()[:, :, ::-1] 

