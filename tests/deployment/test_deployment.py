import os
import boto3
import pytest
import requests
import json
from tests.fixtures import *
from time import time
from typing import Any, Dict, Callable

pytestmark = pytest.mark.deployment

STACK_NAME = os.environ.get("STACK_NAME")
STACK_NAME = "yolo-sam-dev"


@pytest.fixture(scope="session")
def api_gateway_url():
    """Get the API Gateway URL from Cloudformation Stack outputs"""

    if STACK_NAME is None:
        raise ValueError(
            "Please set the AWS_SAM_STACK_NAME environment variable to the name of your stack"
        )

    client = boto3.client("cloudformation")

    try:
        response = client.describe_stacks(StackName=STACK_NAME)
        stacks = response["Stacks"]
        stack_outputs = stacks[0]["Outputs"]
        api_outputs = [
            output for output in stack_outputs if output["OutputKey"] == "DetectionApi"
        ]

        if not api_outputs:
            raise KeyError(f"DetectionApi not found in stack {STACK_NAME}")

    except Exception as e:
        raise Exception(
            f"Cannot find stack {STACK_NAME} \n"
            f'Please make sure a stack with the name "{STACK_NAME}" exists'
        ) from e

    return api_outputs[0]["OutputValue"]  # Extract url from stack outputs


def test_detect_successful_post(
    api_gateway_url: str,
    test_body: dict,
    assert_post_response: Callable[[Dict[str, Any]], None],
):
    start_time = time()
    response = requests.post(
        api_gateway_url, json=test_body, headers={"Content-Type": CONTENT_TYPE_JSON}
    )
    response_time = time() - start_time
    assert_post_response(response.json(), response.status_code)

    # Performance Assertion
    max_allowed_time = 10.0  # seconds
    assert (
        response_time < max_allowed_time
    ), f"Response time exceeded {max_allowed_time} seconds"


def test_detect_successful_get(
    api_gateway_url: str,
    assert_get_response: Callable[[Dict[str, Any]], None],
):
    ret = requests.get(api_gateway_url)
    assert_get_response(ret.json(), ret.status_code)


def test_detect_post_missing_image(
    api_gateway_url: str,
) -> None:
    ret = requests.post(
        api_gateway_url,
        json={"conf_thres": 0.6, "iou_thres": 0.4},
        headers={"Content-Type": CONTENT_TYPE_JSON},
    )
    assert ret.status_code == 400
    body = ret.json()
    assert "error" in body
    assert body["error"] == "Missing 'image' in request body"


@pytest.mark.parametrize(
    "field, value, expected_error",
    [
        ("image", "not_an_image", "Invalid image data. Must be base64 encoded."),
        (
            "conf_thres",
            "not_a_float",
            "Invalid conf_thres. Must be a float between 0 and 1.",
        ),
        (
            "iou_thres",
            "not_a_float",
            "Invalid iou_thres. Must be a float between 0 and 1.",
        ),
        ("conf_thres", 1.5, "conf_thres must be between 0 and 1"),
        ("iou_thres", -0.1, "iou_thres must be between 0 and 1"),
        ("save_image", "not_a_boolean", "Invalid save_image. Must be a boolean."),
    ],
)
def test_detect_post_invalid_fields(
    api_gateway_url: str,
    test_body: dict,
    field: str,
    value: Any,
    expected_error: str,
) -> None:
    body = test_body.copy()
    body[field] = value
    ret = requests.post(
        api_gateway_url, json=body, headers={"Content-Type": "application/json"}
    )
    assert_error_response(ret.json(), ret.status_code, expected_error)
