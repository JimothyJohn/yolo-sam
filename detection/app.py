#!/usr/bin/env python3
import os
import json
import base64
import logging
from io import BytesIO
from typing import Dict, List, Any
from PIL import Image

try:
    # Local development (package-relative imports)
    from .onnx_detector import OnnxDetector, DetectionResult
    from .utils import GET_RESPONSE
    from .coco import CocoClassMapper
    from .config import MODEL_PATH
except ImportError:
    # Lambda environment (flat imports)
    from onnx_detector import OnnxDetector, DetectionResult
    from utils import GET_RESPONSE
    from coco import CocoClassMapper
    from config import MODEL_PATH
import functools
from functools import lru_cache

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of logs

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@lru_cache(maxsize=1)
def get_detector():
    """
    AI-generated comment: Cached initialization of OnnxDetector to improve performance on subsequent invocations.
    """
    model_type = "rf-detr" if "rf-detr" in MODEL_PATH.lower() else "yolov8"
    return OnnxDetector(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), MODEL_PATH),
        model_type=model_type,
    )


@lru_cache(maxsize=1)
def get_class_mapper():
    """
    AI-generated comment: Cached initialization of CocoClassMapper to improve performance on subsequent invocations.
    """
    return CocoClassMapper(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "coco.yaml")
    )


def detect(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Perform object detection on an image and return detections with class names.

    :param body: A dictionary containing the image and detection parameters.
    :return: Detections with class names.
    """
    logger.info("Starting detection process")

    # Get parameters
    img_b64 = body["image"]
    # This size will need to be changed based on the model being used.
    # RF-DETR Nano for example needs 384
    # SIZE = 384
    # Yolov8 Nano needs 640
    SIZE = 640
    conf_thres = body.get("conf_thres", 0.7)
    iou_thres = body.get("iou_thres", 0.5)

    logger.debug(
        f"Detection parameters: SIZE={SIZE}, conf_thres={conf_thres}, iou_thres={iou_thres}"
    )

    # Open and resize image
    try:
        img = Image.open(BytesIO(base64.b64decode(img_b64.encode("ascii"))))
        # Image resizing is handled inside OnnxDetector logic for YOLOv8 (via utils.prepare_input)
        # But wait, utils.prepare_input takes the PIL image.
        # The old app.py resized it MANUALLY before passing to detector:
        # img_resized = img.resize((SIZE, SIZE))
        # Then detector was called with img_resized.
        # My new OnnxDetector._predict_yolov8 calls utils.prepare_input(img, size).
        # utils.prepare_input RESIZES it again based on logic.
        # So I should pass the ORIGINAL image to OnnxDetector?
        # Let's check utils.prepare_input: it takes (img, size), calculates ratio, resizes.
        # So yes, I should pass the PIL image directly.

        logger.debug("Image successfully decoded")
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

    # Infer result
    try:
        detector = get_detector()
        # OnnxDetector expects PIL Image
        detection_result: DetectionResult = detector(
            img, size=SIZE, conf_thres=conf_thres, iou_thres=iou_thres
        )

        # Convert to list of dicts for API response compatibility
        detections = detection_result.to_dict_list()

        logger.info(f"Detection completed. Found {len(detections)} objects.")
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise

    # Map class IDs to names
    class_mapper = get_class_mapper()
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
    # AI-generated comment: Simplified validation using a dictionary of validation rules
    validation_rules = {
        "image": {
            "required": True,
            "validator": lambda x: Image.open(
                BytesIO(base64.b64decode(x.encode("ascii")))
            ),
            "error": "Invalid image data. Must be base64 encoded.",
        },
        "conf_thres": {
            "default": 0.7,
            "validator": lambda x: 0 <= float(x) <= 1,
            "error": "Invalid conf_thres. Must be a float between 0 and 1.",
            "range_error": "conf_thres must be between 0 and 1",
        },
        "iou_thres": {
            "default": 0.5,
            "validator": lambda x: 0 <= float(x) <= 1,
            "error": "Invalid iou_thres. Must be a float between 0 and 1.",
            "range_error": "iou_thres must be between 0 and 1",
        },
        "save_image": {
            "default": False,
            "validator": lambda x: isinstance(x, bool),
            "error": "Invalid save_image. Must be a boolean.",
        },
    }

    validated = {}
    for key, rule in validation_rules.items():
        value = body.get(key, rule.get("default"))
        if rule.get("required", False) and value is None:
            return False, {"error": f"Missing '{key}' in request body"}
        if value is not None:
            try:
                if key in ["conf_thres", "iou_thres"]:
                    float_value = float(value)
                    if not rule["validator"](float_value):
                        return False, {"error": rule["range_error"]}
                    validated[key] = float_value
                elif rule["validator"](value):
                    validated[key] = value
                else:
                    return False, {"error": rule["error"]}
            except ValueError:
                return False, {"error": rule["error"]}
            except Exception:
                return False, {"error": rule["error"]}

    return True, validated


# AI-generated comment: Decorator for error handling and logging
def api_error_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            return {
                "statusCode": 500,
                "body": json.dumps({"message": f"An error occurred: {str(e)}"}),
            }

    return wrapper


@api_error_handler
def lambda_handler(event: dict, context: Any) -> dict:
    logger.info("Lambda function invoked")

    # Define CORS headers
    cors_headers = {
        "Access-Control-Allow-Origin": "*",  # AI-generated comment: Allow requests from any origin. Adjust as needed for security.
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",  # AI-generated comment: Specify allowed HTTP methods.
        "Access-Control-Allow-Headers": "Content-Type",  # AI-generated comment: Specify allowed headers.
    }

    if event.get("httpMethod") == "OPTIONS":
        # AI-generated comment: Handle preflight OPTIONS request for CORS
        return {
            "statusCode": 200,
            "headers": cors_headers,
            "body": json.dumps({"message": "CORS preflight check successful"}),
        }

    if event.get("httpMethod") == "GET":
        return {
            "statusCode": 200,
            "body": json.dumps(GET_RESPONSE),
            "headers": {**cors_headers, "Content-Type": "application/json"},
        }

    try:
        body = json.loads(event.get("body", "{}"))
    except json.JSONDecodeError:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid JSON in request body"}),
            "headers": cors_headers,
        }

    is_valid, validated_body = validate_body(body)
    if not is_valid:
        return {
            "statusCode": 400,
            "body": json.dumps(validated_body),
            "headers": cors_headers,
        }

    detections = detect(validated_body)
    return {
        "statusCode": 200,
        "body": json.dumps({"detections": detections}),
        "headers": cors_headers,
    }
