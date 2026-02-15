import pytest
from tests.utils import encode_image_to_base64
from detection.utils import GET_RESPONSE

CONTENT_TYPE_JSON = "application/json"
HTTP_METHOD_POST = "POST"
HTTP_METHOD_GET = "GET"
STATUS_CODE_OK = 200
STATUS_CODE_BAD_REQUEST = 400


@pytest.fixture(scope="function")
def test_body() -> dict:
    return {
        "image": encode_image_to_base64("images/zidane.jpg"),
        "size": 640,
        "conf_thres": 0.7,
        "iou_thres": 0.5,
        "save_image": True,
    }


# AI-generated comment: Helper function to assert CORS headers
@pytest.fixture(scope="function")
def assert_cors_headers():

    def _assert_cors_headers(headers: dict[str, str]):
        assert "Access-Control-Allow-Origin" in headers
        assert headers["Access-Control-Allow-Origin"] == "*"
        assert "Access-Control-Allow-Methods" in headers
        assert "Access-Control-Allow-Headers" in headers

    return _assert_cors_headers


@pytest.fixture(scope="function")
def apigw_event_factory() -> dict:
    """Factory fixture to generate customizable API Gateway events"""

    def _make_apigw_event(
        body=None,
        resource="/{proxy+}",
        path="/examplepath",
        http_method="GET",
        stage="prod",
        api_key="",
        user="",
        query_params=None,
        headers=None,
        path_params=None,
        stage_vars=None,
    ):
        # AI-generated comment: This function creates a customizable API Gateway event
        # It allows easy modification of common event properties for different test scenarios
        event = {
            "body": body,
            "resource": resource,
            "path": path,
            "httpMethod": http_method,
            "requestContext": {
                "resourceId": "123456",
                "apiId": "1234567890",
                "resourcePath": resource,
                "httpMethod": http_method,
                "requestId": "c6af9ac6-7b61-11e6-9a41-93e8deadbeef",
                "accountId": "123456789012",
                "stage": stage,
                "identity": {
                    "apiKey": api_key,
                    "userArn": "",
                    "cognitoAuthenticationType": "",
                    "caller": "",
                    "userAgent": "Custom User Agent String",
                    "user": user,
                    "cognitoIdentityPoolId": "",
                    "cognitoIdentityId": "",
                    "cognitoAuthenticationProvider": "",
                    "sourceIp": "127.0.0.1",
                    "accountId": "",
                },
            },
            "queryStringParameters": query_params or {},
            "headers": headers
            or {
                "Via": "1.1 08f323deadbeefa7af34d5feb414ce27.cloudfront.net (CloudFront)",
                "Accept-Language": "en-US,en;q=0.8",
                "CloudFront-Is-Desktop-Viewer": "true",
                "CloudFront-Is-SmartTV-Viewer": "false",
                "CloudFront-Is-Mobile-Viewer": "false",
                "X-Forwarded-For": "127.0.0.1, 127.0.0.2",
                "CloudFront-Viewer-Country": "US",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Upgrade-Insecure-Requests": "1",
                "X-Forwarded-Port": "443",
                "Host": "1234567890.execute-api.us-east-1.amazonaws.com",
                "X-Forwarded-Proto": "https",
                "X-Amz-Cf-Id": "aaaaaaaaaae3VYQb9jd-nvCd-de396Uhbp027Y2JvkCPNLmGJHqlaA==",
                "CloudFront-Is-Tablet-Viewer": "false",
                "Cache-Control": "max-age=0",
                "User-Agent": "Custom User Agent String",
                "CloudFront-Forwarded-Proto": "https",
                "Accept-Encoding": "gzip, deflate, sdch",
            },
            "pathParameters": path_params or {"proxy": path.lstrip("/")},
            "stageVariables": stage_vars or {"baz": "qux"},
        }
        return event

    return _make_apigw_event


@pytest.fixture
def assert_get_response():
    def _assert_get_response(body, status_code):
        assert status_code == 200

        assert body["message"] == GET_RESPONSE["message"]
        assert "usage" in body
        assert "description" in body

        usage = body["usage"]
        assert usage["method"] == GET_RESPONSE["usage"]["method"]
        assert usage["content_type"] == GET_RESPONSE["usage"]["content_type"]
        assert "body" in usage

        usage_body = usage["body"]
        assert "image" in usage_body
        assert "conf_thres" in usage_body
        assert "iou_thres" in usage_body

        assert body["description"]

        assert usage_body["image"] == GET_RESPONSE["usage"]["body"]["image"]
        assert usage_body["conf_thres"] == GET_RESPONSE["usage"]["body"]["conf_thres"]
        assert usage_body["iou_thres"] == GET_RESPONSE["usage"]["body"]["iou_thres"]

        assert body["description"] == GET_RESPONSE["description"]

    return _assert_get_response


@pytest.fixture
def assert_post_response():
    def _assert_post_response(body, status_code):
        assert status_code == 200

        assert "detections" in body
        assert isinstance(body["detections"], list)
        assert len(body["detections"]) > 0

        for detection in body["detections"]:
            assert "bbox" in detection
            assert "score" in detection
            assert "class_name" in detection
            assert isinstance(detection["bbox"], list)
            assert isinstance(detection["score"], float)
            assert isinstance(detection["class_name"], str)
            assert len(detection["bbox"]) == 4

    return _assert_post_response


@pytest.fixture(scope="function")
def assert_error_response():
    def _assert_error_response(body, status_code, expected_error) -> None:
        assert status_code == STATUS_CODE_BAD_REQUEST
        assert "error" in body
        assert body["error"] == expected_error

    return _assert_error_response
