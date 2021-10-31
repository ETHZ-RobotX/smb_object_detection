import rospy
import torch

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from rospy.exceptions import ROSInitException 

class ObjectDetector:
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
            
            if self.device == 'cpu':
                raise rospy.exceptions.ROSInitException("Detectron architecture runs very slow on CPU. Not recommended.")

            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file(self.model)) # "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            self.checkpoint  = self.model
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.checkpoint)
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.iou 
            self.cfg.MODEL.DEVICE = self.device
            self.detector = DefaultPredictor(self.cfg)
        else:
            raise rospy.exceptions.ROSInitException("Unrecognised architecture.")

    def detect(self, image, return_image = False):
        """ Detects objects on the given image using set model (YOLO V5 or Detectron) 
        
        Args:
            image        : numpy matrix, RGB or Gray Scale image
            return_image : bool to decide return image or not 
        
        Returns:
            A list  -> list[0] : object infos in Pandas data frame
                    -> list[1] : image with bounding boxes

        """

        if self.architecture == 'yolo':
            output = self.detector(image)
            if return_image:
                return [output.pandas().xyxy[0], output.render()[0] ]
            
            else:
                return [output.pandas().xyxy[0], None]
            

        elif self.architecture == 'detectron':
            outputs = self.detector(image)
            v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            return out.get_image()[:, :, ::-1]