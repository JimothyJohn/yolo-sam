import pytest
import os
import psutil
import gc
from PIL import Image
from detection.onnx_detector import OnnxDetector

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YOLO_PATH = os.path.join(BASE_DIR, "detection", "models", "yolov8n.onnx")
RF_DETR_PATH = os.path.join(BASE_DIR, "detection", "models", "rf-detr-nano.onnx")


@pytest.fixture(scope="session")
def sample_image():
    # create a dummy image comparable to real usage (e.g. 640x480)
    return Image.new("RGB", (640, 480), color="blue")


def measure_peak_memory(func, *args):
    """
    Measures RSS memory increase during function execution.
    Note: accurate generic memory measurement is hard. This gives a rough delta.
    """
    pid = os.getpid()
    process = psutil.Process(pid)

    # Force gc before
    gc.collect()
    mem_before = process.memory_info().rss / 1024 / 1024

    # Run
    func(*args)

    # Check peak? No, psutil only gives current.
    # For a true peak, we'd need a background thread sampler,
    # but for model inference, the steady state after loading + running is what matters most for Lambda.
    # So we simply return the delta (which includes model weights loaded into RAM).
    mem_after = process.memory_info().rss / 1024 / 1024

    return mem_after - mem_before, mem_after


@pytest.mark.benchmark(group="inference_time")
def test_speed_yolov8(benchmark, sample_image):
    if not os.path.exists(YOLO_PATH):
        pytest.skip("YOLO model not found")

    detector = OnnxDetector(YOLO_PATH, model_type="yolov8")

    # Warmup
    detector(sample_image)

    # Benchmark
    benchmark(detector, sample_image)


@pytest.mark.benchmark(group="inference_time")
def test_speed_rf_detr(benchmark, sample_image):
    if not os.path.exists(RF_DETR_PATH):
        pytest.skip("RF-DETR model not found")

    detector = OnnxDetector(RF_DETR_PATH, model_type="rf-detr")

    # Warmup
    detector(sample_image)

    # Benchmark
    benchmark(detector, sample_image)


@pytest.mark.performance
def test_memory_yolov8(sample_image):
    if not os.path.exists(YOLO_PATH):
        pytest.skip("YOLO model not found")

    process = psutil.Process(os.getpid())
    gc.collect()
    mem_start = process.memory_info().rss / 1024 / 1024

    detector = OnnxDetector(YOLO_PATH, model_type="yolov8")
    detector(sample_image)  # Inference to ensure full allocation

    mem_end = process.memory_info().rss / 1024 / 1024
    delta = mem_end - mem_start
    print(f"\n[YOLOv8] Memory Delta: {delta:.2f} MB | Total: {mem_end:.2f} MB")

    # Assert sensible limits (e.g., < 500MB for Nano)
    assert delta < 500, "YOLOv8 model uses excessive memory"


@pytest.mark.performance
def test_memory_rf_detr(sample_image):
    if not os.path.exists(RF_DETR_PATH):
        pytest.skip("RF-DETR model not found")

    process = psutil.Process(os.getpid())
    gc.collect()
    mem_start = process.memory_info().rss / 1024 / 1024

    detector = OnnxDetector(RF_DETR_PATH, model_type="rf-detr")
    detector(sample_image)  # Inference

    mem_end = process.memory_info().rss / 1024 / 1024
    delta = mem_end - mem_start
    print(f"\n[RF-DETR] Memory Delta: {delta:.2f} MB | Total: {mem_end:.2f} MB")

    # RF-DETR might use more, but checking reasonable bounds
    assert delta < 1000, "RF-DETR model uses excessive memory"
