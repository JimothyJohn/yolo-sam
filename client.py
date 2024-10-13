# AI-generated comment: This script demonstrates how to use the YOLOv8 detection API
# and visualize the results by drawing bounding boxes on the original image.
# It now includes a CLI interface for easier use and parameter adjustment.

import base64
import requests
from PIL import Image, ImageDraw, ImageFont
import argparse

# AI-generated comment: Constants for API endpoint
API_ENDPOINT = "https://yolo.advin.io/Prod/detect"


def encode_image(image_path):
    """
    AI-generated comment: Encodes the image file to base64 for API request
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def send_request(encoded_image, conf_thres, iou_thres):
    """
    AI-generated comment: Sends a POST request to the API with the encoded image
    """
    payload = {
        "image": encoded_image,
        "conf_thres": conf_thres,
        "iou_thres": iou_thres,
    }
    response = requests.post(API_ENDPOINT, json=payload)
    print(response.json())
    return response.json()


def draw_boxes(image_path, detections):
    """
    AI-generated comment: Draws bounding boxes and labels on the image
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # AI-generated comment: Attempt to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 64)
    except IOError:
        font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label = f"{detection['class_name']} {detection['score']:.2f}"

        # AI-generated comment: Draw green bounding box
        draw.rectangle([x1, y1, x2, y2], outline="green", width=8)

        # AI-generated comment: Draw label background
        text_bbox = draw.textbbox((x1, y1), label)
        draw.rectangle(text_bbox, fill="green")

        # AI-generated comment: Draw white text
        draw.text((x1, y1), label, fill="white", font=font)

    return image


def main(args):
    # AI-generated comment: Encode the image
    encoded_image = encode_image(args.input)

    # AI-generated comment: Send request to API
    response = send_request(encoded_image, args.conf_thres, args.iou_thres)

    if "detections" in response:
        # AI-generated comment: Draw bounding boxes on the image
        output_image = draw_boxes(args.input, response["detections"])

        # AI-generated comment: Save the output image
        output_image.save(args.output)
        print(f"Output image saved as {args.output}")
    else:
        print("Error: No detections found in the API response")


if __name__ == "__main__":
    # AI-generated comment: Set up argument parser for CLI
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection CLI")
    parser.add_argument("-i", "--input", required=True, help="Path to input image")
    parser.add_argument(
        "-o",
        "--output",
        default="output_image.jpg",
        help="Path to output image (default: output_image.jpg)",
    )
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--iou_thres", type=float, default=0.5, help="IoU threshold (default: 0.5)"
    )

    args = parser.parse_args()

    main(args)
