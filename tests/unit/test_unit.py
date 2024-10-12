import json
import pytest
from detection.app import lambda_handler, detect, validate_body, api_error_handler
from tests.fixtures import *
import pytest
import json
from typing import Callable, Dict, Any
from unittest.mock import patch


pytestmark = pytest.mark.unit


def test_detect_successful_post(
    apigw_event_factory: Callable[..., Dict[str, Any]],
    test_body: dict,
    assert_post_response: Callable[[Dict[str, Any]], None],
) -> None:
    event = apigw_event_factory(
        http_method="POST",
        body=json.dumps(test_body),
        headers={"Content-Type": "application/json"},
    )

    ret = lambda_handler(event, "")
    assert_post_response(json.loads(ret["body"]), ret["statusCode"])


def test_detect_successful_get(
    apigw_event_factory: Callable[..., Dict[str, Any]],
    assert_get_response: Callable[[Dict[str, Any]], None],
) -> None:
    # This test checks the response for a GET request,
    # which should return a 200 status code with information about using POST requests
    event = apigw_event_factory(
        body="",
        http_method="GET",
    )
    ret = lambda_handler(event, "")
    assert_get_response(json.loads(ret["body"]), ret["statusCode"])


def test_detect_post_missing_image(
    apigw_event_factory: Callable[..., Dict[str, Any]]
) -> None:
    event = apigw_event_factory(
        http_method="POST",
        body=json.dumps({"conf_thres": 0.6, "iou_thres": 0.4}),
        headers={"Content-Type": "application/json"},
    )

    ret = lambda_handler(event, "")
    assert ret["statusCode"] == 400
    body = json.loads(ret["body"])
    assert "error" in body
    assert body["error"] == "Missing 'image' in request body"


def test_detect_post_valid_save_image(apigw_event_factory, test_body: dict) -> None:
    body = test_body.copy()
    body["save_image"] = True
    event = apigw_event_factory(
        http_method="POST",
        body=json.dumps(body),
        headers={"Content-Type": "application/json"},
    )

    ret = lambda_handler(event, "")
    assert ret["statusCode"] == 200
    body = json.loads(ret["body"])
    assert "detections" in body


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
    apigw_event_factory: Callable[..., Dict[str, Any]],
    test_body: dict,
    field: str,
    value: Any,
    expected_error: str,
) -> None:
    body = test_body
    body[field] = value
    event = apigw_event_factory(
        http_method="POST",
        body=json.dumps(body),
        headers={"Content-Type": "application/json"},
    )

    ret = lambda_handler(event, "")
    assert_error_response(json.loads(ret["body"]), ret["statusCode"], expected_error)


import json
import pytest
from detection.app import lambda_handler, validate_body, detect
from tests.fixtures import *
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.unit


def test_detect_successful_post(
    apigw_event_factory: Callable[..., Dict[str, Any]],
    test_body: dict,
    assert_post_response: Callable[[Dict[str, Any]], None],
) -> None:
    event = apigw_event_factory(
        http_method="POST",
        body=json.dumps(test_body),
        headers={"Content-Type": "application/json"},
    )

    ret = lambda_handler(event, "")
    assert_post_response(json.loads(ret["body"]), ret["statusCode"])


def test_detect_successful_get(
    apigw_event_factory: Callable[..., Dict[str, Any]],
    assert_get_response: Callable[[Dict[str, Any]], None],
) -> None:
    event = apigw_event_factory(
        body="",
        http_method="GET",
    )
    ret = lambda_handler(event, "")
    assert_get_response(json.loads(ret["body"]), ret["statusCode"])


def test_detect_post_missing_image(
    apigw_event_factory: Callable[..., Dict[str, Any]]
) -> None:
    event = apigw_event_factory(
        http_method="POST",
        body=json.dumps({"conf_thres": 0.6, "iou_thres": 0.4}),
        headers={"Content-Type": "application/json"},
    )

    ret = lambda_handler(event, "")
    assert ret["statusCode"] == 400
    body = json.loads(ret["body"])
    assert "error" in body
    assert body["error"] == "Missing 'image' in request body"


def test_detect_post_valid_save_image(apigw_event_factory, test_body: dict) -> None:
    body = test_body.copy()
    body["save_image"] = True
    event = apigw_event_factory(
        http_method="POST",
        body=json.dumps(body),
        headers={"Content-Type": "application/json"},
    )

    ret = lambda_handler(event, "")
    assert ret["statusCode"] == 200
    body = json.loads(ret["body"])
    assert "detections" in body


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
    apigw_event_factory: Callable[..., Dict[str, Any]],
    test_body: dict,
    field: str,
    value: Any,
    expected_error: str,
) -> None:
    body = test_body.copy()
    body[field] = value
    event = apigw_event_factory(
        http_method="POST",
        body=json.dumps(body),
        headers={"Content-Type": "application/json"},
    )

    ret = lambda_handler(event, "")
    assert_error_response(json.loads(ret["body"]), ret["statusCode"], expected_error)


def test_invalid_json_body(apigw_event_factory):
    event = apigw_event_factory(
        http_method="POST",
        body="invalid json",
        headers={"Content-Type": "application/json"},
    )

    ret = lambda_handler(event, "")
    assert ret["statusCode"] == 400
    body = json.loads(ret["body"])
    assert "error" in body
    assert body["error"] == "Invalid JSON in request body"


def test_api_error_handler():
    @api_error_handler
    def mock_function():
        raise ValueError("Test error")

    result = mock_function()
    assert result["statusCode"] == 500
    assert "An error occurred: Test error" in result["body"]
