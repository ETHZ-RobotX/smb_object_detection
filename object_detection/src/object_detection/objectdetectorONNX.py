import os
import pandas as pd
import numpy as np
import onnxruntime as rt
import cv2  # Used for preprocessing and postprocessing using OpenCV

class ObjectDetectorONNX:
    def __init__(self, config):
        self.architecture = config["architecture"]
        self.model = config["model"]
        self.model_dir_path = config["model_dir_path"]
        self.checkpoint = config["checkpoint"]
        self.device = config["device"]
        self.confident = config["confident"]
        self.iou = config["iou"]
        self.classes = config["classes"]
        self.multiple_instance = config["multiple_instance"]
        self.detector = None

        if self.architecture == 'yolo':
            if self.model_dir_path:
                onnx_model_path = os.path.join(self.model_dir_path, self.model + ".onnx")
                print(f"Loading ONNX model from: {onnx_model_path}")
                self.session = rt.InferenceSession(onnx_model_path)
                self.input_name = self.session.get_inputs()[0].name
            else:
                raise ValueError("No model path defined for ONNX model.")
        elif self.architecture == 'detectron':
            raise ValueError("Detectron return type was not adapted. Implement it if needed.")
        else:
            raise ValueError("Unrecognised architecture.")
    
    def preprocess(self, image):
        # Maintain aspect ratio and pad the image to the required input size
        input_size = 640
        h, w, _ = image.shape
        scale = min(input_size / h, input_size / w)
        nh, nw = int(h * scale), int(w * scale)
        image_resized = cv2.resize(image, (nw, nh))
        
        top = (input_size - nh) // 2
        bottom = input_size - nh - top
        left = (input_size - nw) // 2
        right = input_size - nw - left
        
        image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        image_padded = image_padded.astype(np.float16) / 255.0
        image_padded = np.transpose(image_padded, (2, 0, 1))  # Change to (C, H, W)
        image_padded = np.expand_dims(image_padded, axis=0)  # Add batch dimension
        return image_padded, scale, top, left

    
    confident = 0.5
    class_dict = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
                5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
                14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
                30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
                35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
                45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
                51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
                57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
                62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
                68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
                73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                78: 'hair drier', 79: 'toothbrush'}
                 
    def postprocess(self, detection, original_width, original_height, scale, pad_top, pad_left, input_size=1280, conf_threshold=0.5):
        try:
            # Ensure detection is a numpy array
            if isinstance(detection, list):
                detection = detection[0]

            detection = detection[0]

            if detection.shape[1] != 85:
                raise ValueError("Detection tensor shape is incorrect.")

            boxes = detection[:, :4]  # Bounding boxes (cx, cy, w, h)
            confidences = detection[:, 4]  # Confidence scores
            class_probs = detection[:, 5:]  # Class probabilities

            # Filter out low confidence detections
            indices = np.where(confidences > conf_threshold)

            boxes = boxes[indices]
            confidences = confidences[indices]
            class_probs = class_probs[indices]

            # Get class indices
            class_indices = np.argmax(class_probs, axis=1)

            # Convert center coordinates to corner coordinates
            cx = boxes[:, 0]
            cy = boxes[:, 1]
            w = boxes[:, 2]
            h = boxes[:, 3]
            
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # Scale boxes back to the original image size and correct for padding
            x1 = (x1 - pad_left) / scale
            y1 = (y1 - pad_top) / scale
            x2 = (x2 - pad_left) / scale
            y2 = (y2 - pad_top) / scale

            # Ensure the coordinates are within image dimensions
            x1 = np.clip(x1, 0, original_width)
            y1 = np.clip(y1, 0, original_height)
            x2 = np.clip(x2, 0, original_width)
            y2 = np.clip(y2, 0, original_height)

            # Map class indices to class names
            filtered_names = [self.class_dict[c] for c in class_indices]

            result_df = pd.DataFrame({
                'xmin': x1,
                'ymin': y1,
                'xmax': x2,
                'ymax': y2,
                'confidence': confidences,
                'class': class_indices,
                'name': filtered_names
            })

            return result_df

        except Exception as e:
            print(f"An error occurred: {e}")
            raise




    def detect(self, image):
        if self.architecture == 'yolo':
            original_height, original_width = image.shape[:2]
            input_image, scale, pad_top, pad_left = self.preprocess(image)
            outputs = self.session.run(None, {self.input_name: input_image})

            detection = self.postprocess(outputs, original_width, original_height, scale, pad_top, pad_left)

            if not self.multiple_instance:
                detection = self.filter_detection(detection)

            # Drawing the bounding boxes on the image
            for index, row in detection.iterrows():
                cv2.rectangle(image, (int(row['xmin']), int(row['ymin'])), 
                            (int(row['xmax']), int(row['ymax'])), 
                            (0, 255, 0), 2)
                cv2.putText(image, f"{row['name']} {row['confidence']:.2f}", 
                            (int(row['xmin']), int(row['ymin']) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            return detection, image



        
    def filter_detection(self, detection):
        detected_objects = []
        row_to_delete = []
        for i in range(len(detection)):
            if detection['class'][i] in detected_objects:
                row_to_delete.append(i)
            else:
                detected_objects.append(detection['class'][i])
        
        detection = detection.drop(row_to_delete, axis=0)
        detection.reset_index(inplace=True, drop=True)

        return detection