import os
import pytest
import requests
import json
from tests.fixtures import *
from time import time
from typing import Any, Dict, Callable

pytestmark = pytest.mark.integration

API_GATEWAY_URL = os.environ.get("API_GATEWAY_URL")


def test_detect_successful_post(
    test_body: dict,
    assert_post_response: Callable[[Dict[str, Any]], None],
    assert_cors_headers: Callable[[dict[str, str]], None],
):
    start_time = time()
    response = requests.post(
        url=API_GATEWAY_URL, json=test_body, headers={"Content-Type": CONTENT_TYPE_JSON}
    )
    response_time = time() - start_time
    assert_post_response(response.json(), response.status_code)
    assert_cors_headers(response.headers)

    # Performance Assertion
    max_allowed_time = 10.0  # seconds
    assert (
        response_time < max_allowed_time
    ), f"Response time exceeded {max_allowed_time} seconds"


def test_detect_successful_get(
    assert_get_response: Callable[[Dict[str, Any]], None],
    assert_cors_headers: Callable[[dict[str, str]], None],
):
    response = requests.get(API_GATEWAY_URL)
    assert_get_response(response.json(), response.status_code)
    assert_cors_headers(response.headers)


def test_detect_post_missing_image(
    assert_cors_headers: Callable[[dict[str, str]], None],
) -> None:
    response = requests.post(
        url=API_GATEWAY_URL,
        json={"conf_thres": 0.6, "iou_thres": 0.4},
        headers={"Content-Type": CONTENT_TYPE_JSON},
    )
    assert response.status_code == 400
    body = response.json()
    assert "error" in body
    assert body["error"] == "Missing 'image' in request body"
    assert_cors_headers(response.headers)


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
    test_body: dict,
    field: str,
    value: Any,
    expected_error: str,
    assert_error_response: Callable[[Dict[str, Any]], None],
    assert_cors_headers: Callable[[dict[str, str]], None],
) -> None:
    body = test_body.copy()
    body[field] = value
    response = requests.post(
        url=API_GATEWAY_URL, json=body, headers={"Content-Type": "application/json"}
    )
    assert_error_response(response.json(), response.status_code, expected_error)
    assert_cors_headers(response.headers)
