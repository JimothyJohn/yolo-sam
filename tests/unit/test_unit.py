import json
import pytest
from detection.app import lambda_handler, api_error_handler
from typing import Callable, Any

pytestmark = pytest.mark.unit


def test_detect_successful_post(
    apigw_event_factory: Callable[..., dict[str, Any]],
    test_body: dict,
    assert_post_response: Callable[[dict[str, Any]], None],
    assert_cors_headers: Callable[[dict[str, str]], None],
) -> None:
    event = apigw_event_factory(
        http_method="POST",
        body=json.dumps(test_body),
        headers={"Content-Type": "application/json"},
    )

    ret = lambda_handler(event, "")
    assert_post_response(json.loads(ret["body"]), ret["statusCode"])
    assert_cors_headers(ret["headers"])  # Check CORS headers


def test_detect_successful_get(
    apigw_event_factory: Callable[..., dict[str, Any]],
    assert_get_response: Callable[[dict[str, Any]], None],
    assert_cors_headers: Callable[[dict[str, str]], None],
) -> None:
    # This test checks the response for a GET request,
    # which should return a 200 status code with information about using POST requests
    event = apigw_event_factory(
        body="",
        http_method="GET",
    )
    ret = lambda_handler(event, "")
    assert_get_response(json.loads(ret["body"]), ret["statusCode"])
    assert_cors_headers(ret["headers"])  # Check CORS headers


def test_detect_post_missing_image(
    apigw_event_factory: Callable[..., dict[str, Any]],
    assert_cors_headers: Callable[[dict[str, str]], None],
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
    assert_cors_headers(ret["headers"])  # Check CORS headers


def test_detect_post_valid_save_image(
    apigw_event_factory,
    test_body: dict,
    assert_cors_headers: Callable[[dict[str, str]], None],
    monkeypatch,
) -> None:
    """
    Tests the ability to save the sent image to an AWS S3 bucket called ophalim-images
    """
    # Mock S3 client
    import boto3
    from botocore.stub import Stubber, ANY

    # Create a dummy client and stub it
    s3 = boto3.client("s3")
    stubber = Stubber(s3)

    # Expect put_object for image
    stubber.add_response(
        "put_object",
        {},
        {
            "Bucket": "test-bucket",
            "Key": "images/mock-uuid.jpg",
            "Body": ANY,
            "ContentType": "image/jpeg",
        },
    )
    # Expect put_object for json
    stubber.add_response(
        "put_object",
        {},
        {
            "Bucket": "test-bucket",
            "Key": "detections/mock-uuid.json",
            "Body": ANY,
            "ContentType": "application/json",
        },
    )

    # Mock the s3_client in app.py
    # We need to patch the s3_client instance in the module
    import detection.app

    monkeypatch.setattr(detection.app, "s3_client", s3)
    monkeypatch.setattr(detection.app, "IMAGES_BUCKET_NAME", "test-bucket")

    # Mock uuid to return a fixed value
    class MockUUID:
        def __str__(self):
            return "mock-uuid"

    monkeypatch.setattr(detection.app.uuid, "uuid4", MockUUID)

    # Allow the stubber to be used
    with stubber:
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
        assert body["version"] == "0.1"
        assert body["image_url"] == "s3://test-bucket/images/mock-uuid.jpg"
        assert_cors_headers(ret["headers"])  # Check CORS headers


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
    apigw_event_factory: Callable[..., dict[str, Any]],
    assert_cors_headers: Callable[[dict[str, str]], None],
    assert_error_response: Callable[[dict[str, Any]], None],
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
    assert_cors_headers(ret["headers"])  # Check CORS headers


def test_invalid_json_body(
    apigw_event_factory,
    assert_cors_headers: Callable[[dict[str, str]], None],
):
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
    assert_cors_headers(ret["headers"])  # Check CORS headers


def test_api_error_handler():
    @api_error_handler
    def mock_function():
        raise ValueError("Test error")

    result = mock_function()
    assert result["statusCode"] == 500
    assert "An error occurred: Test error" in result["body"]
