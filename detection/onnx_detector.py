from dataclasses import dataclass
from typing import Optional, List, Union, Tuple
import numpy as np
import onnxruntime as ort
import os

import io
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import detection.utils as utils
except ImportError:
    import utils


@dataclass
class DetectionResult:
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    masks: Optional[np.ndarray] = None

    # Optional: Helper to convert back to list of dicts for API compatibility
    def to_dict_list(self) -> List[dict]:
        detections = []
        for i in range(len(self.scores)):
            det = {
                "bbox": [int(x) for x in self.boxes[i]],
                "score": float(self.scores[i]),
                "class_id": int(self.labels[i]),
            }
            # Add mask if needed, though usually too heavy for basic JSON API without encoding
            detections.append(det)
        return detections


class OnnxDetector:
    """
    Unified ONNX Detector for YOLOv8 and RF-DETR models.
    """

    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]

    def __init__(
        self, model_path: str, model_type: str = "yolov8", device: str = "cpu"
    ):
        self.model_path = model_path
        self.model_type = model_type.lower()

        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "gpu":
            providers = ["CUDAExecutionProvider"]
        else:
            raise ValueError(f"Device {device} is not available.")

        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            # Get input shape
            input_info = self.session.get_inputs()[0]
            # Handle dynamic shapes or fixed shapes
            # YOLOv8 often has specific input requirements handled in utils.py
            # RF-DETR has shape attribute
            if hasattr(input_info, "shape") and len(input_info.shape) >= 4:
                self.input_height, self.input_width = input_info.shape[2:]
            else:
                # Default fallback or dynamic
                self.input_height, self.input_width = 640, 640

        except Exception as e:
            raise RuntimeError(
                f"Failed to load ONNX model from '{model_path}': {e}"
            ) from e

    def __call__(
        self,
        image: Image.Image,
        size: int = 640,
        conf_thres: float = 0.7,
        iou_thres: float = 0.5,
    ) -> DetectionResult:
        """
        Main entry point for inference, unifying the call signature.
        """
        return self.predict(image, size, conf_thres, iou_thres)

    def predict(
        self,
        image: Image.Image,
        size: int = 640,
        conf_thres: float = 0.7,
        iou_thres: float = 0.5,
    ) -> DetectionResult:

        if self.model_type == "yolov8":
            return self._predict_yolov8(image, size, conf_thres, iou_thres)
        elif self.model_type == "rf-detr":
            return self._predict_rfdetr(
                image, conf_thres, 300
            )  # RF-DETR often uses fixed 300 boxes
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _predict_yolov8(
        self, img: Image.Image, size: int, conf_thres: float, iou_thres: float
    ) -> DetectionResult:
        # Prepare input with forced square padding for fixed-size models
        inp, orig_size, scaled_size = self._prepare_input(
            img, self.input_width, self.input_height
        )

        # Inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(["output0"], {input_name: inp})

        # Post-process
        boxes, scores, class_ids = utils.post_process(
            outputs, conf_thres=conf_thres, iou_thres=iou_thres
        )

        # Scale boxes
        boxes = utils.scale_boxes(boxes, orig_size, scaled_size)

        return DetectionResult(
            boxes=boxes,
            scores=scores,
            labels=class_ids,
            masks=None,  # YOLOv8n detection model doesn't output masks by default
        )

    def _prepare_input(
        self, img: Image.Image, target_w: int, target_h: int
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Prepares input image for ONNX model with strict target size padding.
        """
        orig_w, orig_h = img.size

        # Calculate scale ratio to fit within target dimensions
        ratio = min(target_w / orig_w, target_h / orig_h)

        # New size
        scaled_w = int(round(orig_w * ratio))
        scaled_h = int(round(orig_h * ratio))

        # Resize
        scaled_img = img.resize(
            (scaled_w, scaled_h), resample=Image.Resampling.BILINEAR
        )

        # Prepare tensor with padding
        # Fill with gray (114) for padding
        inp = np.full((target_h, target_w, 3), 114, dtype=np.float32)

        # Paste scaled image (top-left alignment)
        # Note: YOLO often uses centered padding, but utils.scale_boxes logic (and many exports)
        # assume top-left if using the simple scale_boxes logic from utils.py.
        # We stick to top-left to be compatible with utils.scale_boxes.
        npy_img = np.array(scaled_img)
        inp[:scaled_h, :scaled_w, :] = npy_img

        # Scale to 0-1 and transpose to CHW
        inp = inp / 255.0
        inp = inp.transpose(2, 0, 1)
        inp = np.expand_dims(inp, axis=0)  # Batch dimension

        return inp, (orig_w, orig_h), (scaled_w, scaled_h)

    def _predict_rfdetr(
        self, image: Image.Image, conf_thres: float, max_number_boxes: int = 300
    ) -> DetectionResult:
        # Preprocess
        inp, orig_size = self._prepare_input_rfdetr(
            image, self.input_width, self.input_height
        )

        # Inference (input names might vary, but 'input' is standard for RF-DETR export)
        # We can dynamically get the input name from the session to be safe
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: inp})

        # Post-process
        # outputs[0] = pred_boxes (1, 300, 4)
        # outputs[1] = pred_logits (1, 300, 91) (before sigmoid)

        pred_boxes = outputs[0]
        pred_logits = outputs[1]

        # Apply sigmoid
        prob = 1 / (1 + np.exp(-pred_logits))

        # Get scores and labels
        # shape (1, 300, 91) -> (300, 91)
        prob = prob[0]
        scores = np.max(prob, axis=1)
        labels = np.argmax(prob, axis=1)

        # Sort by confidence
        sorted_idx = np.argsort(scores)[::-1]
        scores = scores[sorted_idx][:max_number_boxes]
        labels = labels[sorted_idx][:max_number_boxes]
        boxes = pred_boxes[0][sorted_idx][:max_number_boxes]

        # Filter by threshold
        keep = scores > conf_thres
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]

        # Convert boxes: cxcywh (0-1) -> xyxy (absolute)
        w, h = orig_size
        boxes = self._box_cxcywh_to_xyxy(boxes, w, h)

        return DetectionResult(boxes=boxes, scores=scores, labels=labels, masks=None)

    def _prepare_input_rfdetr(
        self, img: Image.Image, target_w: int, target_h: int
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        orig_w, orig_h = img.size

        # Resize
        img_resized = img.resize((target_w, target_h))

        # Normalize and Standardize
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        img_np = (img_np - self.MEANS) / self.STDS

        # HWC -> CHW
        img_np = img_np.transpose(2, 0, 1)

        # Batch dimension
        img_np = np.expand_dims(img_np, axis=0)

        return img_np.astype(np.float32), (orig_w, orig_h)

    @staticmethod
    def _box_cxcywh_to_xyxy(x, w_img, h_img):
        cx, cy, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        xmin = cx - w / 2
        ymin = cy - h / 2
        xmax = cx + w / 2
        ymax = cy + h / 2

        # Scale to image
        return np.stack(
            [xmin * w_img, ymin * h_img, xmax * w_img, ymax * h_img], axis=-1
        )
