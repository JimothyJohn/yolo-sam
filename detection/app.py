# https://github.com/trainyolo/YOLOv8-aws-lambda/blob/main/lambda-codebase/app.py
import os
import json
import base64
import logging
from io import BytesIO
from typing import Dict, List, Any
from PIL import Image
import yaml
from yolov8_onnx import YOLOv8
from utils import GET_RESPONSE

# AI-generated comment: Set up logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)


class CocoClassMapper:
    def __init__(self, coco_yaml_path: str):
        self.class_id_to_name = self._load_class_names(coco_yaml_path)

    @staticmethod
    def _load_class_names(yaml_path: str) -> Dict[int, str]:
        """
        Loads the class ID to name mapping from the coco.yaml file.

        :param yaml_path: Path to the coco.yaml file.
        :return: A dictionary mapping class IDs to their names.
        """
        try:
            with open(yaml_path, "r") as file:
                data = yaml.safe_load(file)
                return {int(k): v for k, v in data["names"].items()}
        except Exception as e:
            logger.error(f"Error loading class names from {yaml_path}: {e}")
            raise

    def map_class_names(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Maps class IDs in the detections to their corresponding class names.

        :param detections: A list of detection dictionaries.
        :return: The enriched list of detections with class names.
        """
        for detection in detections:
            class_id = detection.get("class_id")
            detection["class_name"] = self.class_id_to_name.get(class_id, "Unknown")
        return detections


yolov8_detector = YOLOv8(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "yolov8n.onnx")
)

mapper = CocoClassMapper(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "coco.yaml")
)


def detect(body: Dict[str, Any], class_mapper: CocoClassMapper) -> Dict[str, Any]:
    """
    Perform object detection on an image and return detections with class names.

    :param body: A dictionary containing the image and detection parameters.
    :param class_mapper: An instance of CocoClassMapper to map class IDs to names.
    :return: Detections with class names.
    """
    logger.info("Starting detection process")

    # Get parameters
    img_b64 = body["image"]
    SIZE = 640
    conf_thres = body.get("conf_thres", 0.7)
    iou_thres = body.get("iou_thres", 0.5)

    logger.debug(
        f"Detection parameters: SIZE={SIZE}, conf_thres={conf_thres}, iou_thres={iou_thres}"
    )

    # Open and resize image
    try:
        img = Image.open(BytesIO(base64.b64decode(img_b64.encode("ascii"))))
        img_resized = img.resize((SIZE, SIZE))
        logger.debug("Image successfully decoded and resized")
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

    # Infer result
    try:
        detections = yolov8_detector(
            img_resized, size=SIZE, conf_thres=conf_thres, iou_thres=iou_thres
        )
        logger.info(f"Detection completed. Found {len(detections)} objects.")
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise

    # Map class IDs to names
    detections_with_names = class_mapper.map_class_names(detections)
    logger.debug("Class names mapped to detections")

    return detections_with_names


def validate_body(body: dict) -> tuple:
    """
    Validates and sanitizes the request body, providing default values where necessary.

    :param body: The request body as a dictionary
    :return: A tuple (is_valid, result) where is_valid is a boolean and result is either
             a sanitized dictionary with validated and default values or an error message
    """
    validated = {}

    # Validate and process 'image'
    if "image" not in body:
        return False, {"error": "Missing 'image' in request body"}
    try:
        # Attempt to decode the image to verify it's valid base64
        # TODO do this in a faster way?
        Image.open(BytesIO(base64.b64decode(body["image"].encode("ascii"))))
        validated["image"] = body["image"]
    except Exception:
        return False, {"error": "Invalid image data. Must be base64 encoded."}

    # Validate and process 'conf_thres'
    conf_thres = body.get("conf_thres", 0.7)
    try:
        conf_thres = float(conf_thres)
        if not 0 <= conf_thres <= 1:
            return False, {"error": "conf_thres must be between 0 and 1"}
        validated["conf_thres"] = conf_thres
    except ValueError:
        return False, {"error": "Invalid conf_thres. Must be a float between 0 and 1."}

    # Validate and process 'iou_thres'
    iou_thres = body.get("iou_thres", 0.5)
    try:
        iou_thres = float(iou_thres)
        if not 0 <= iou_thres <= 1:
            return False, {"error": "iou_thres must be between 0 and 1"}
        validated["iou_thres"] = iou_thres
    except ValueError:
        return False, {"error": "Invalid iou_thres. Must be a float between 0 and 1."}

    # Validate and process 'save_image'
    save_image = body.get("save_image", False)
    if not isinstance(save_image, bool):
        return False, {"error": "Invalid save_image. Must be a boolean."}
    validated["save_image"] = save_image

    return True, validated


def lambda_handler(event, context):
    logger.info("Lambda function invoked")
    logger.debug(f"Received event: {json.dumps(event)}")

    # Check if the event is coming from API Gateway and provide detailed information
    if event["httpMethod"] == "GET":
        return {
            "statusCode": 200,
            "body": json.dumps(GET_RESPONSE),
            "headers": {"Content-Type": "application/json"},
        }

    # Extract the body from the API Gateway event
    try:
        body = json.loads(event["body"])
    except json.JSONDecodeError:
        logger.error("Invalid JSON in request body")
        return {
            "statusCode": 400,
            "body": json.dumps({"message": "Invalid JSON in request body"}),
        }
    except Exception as e:
        logger.error(f"Error during detection process: {e}", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"message": str(e)})}

    is_valid, validated_body = validate_body(body)
    if not is_valid:
        logger.error(f"Validation error: {validated_body['error']}")
        return {"statusCode": 400, "body": json.dumps(validated_body)}

    try:
        detections = detect(validated_body, mapper)
        logger.info("Detection process completed successfully")
        return {"statusCode": 200, "body": json.dumps({"detections": detections})}
    except Exception as e:
        logger.error(f"Error during detection process: {e}", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"message": str(e)})}
