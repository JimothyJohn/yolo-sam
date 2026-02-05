![yolo](docs/yolo.jpg)

# $0bject Detection

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PcYQoKgmpfmB7bsmn2QXFtu3CLRVbJ7N?usp=sharing)

A cost-effective way to detect objects in your environment! Each query completes in about a second and costs fractions of a penny.

**You can run a THOUSAND detections for only TWO cents!**

## Quickstart

```bash
# Test the demo endpoint using the sample CLI
pip install requests Pillow
python client.py -i ./path/to/image.jpg
# Writes results to output_image.jpg

# OR create your own version

# Export nano model
bash ./utils/ExportONNX.sh
# Format, validate, build, and test
bash ./Quickstart -t
```

### Powered by:

* [AWS SAM](https://aws.amazon.com/serverless/sam/): Automated deployment with a single command. Accessible anyime, anywhere!

* [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime): Accelerated CPU runtime. No expensive GPUs!

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics): Pretrained, general purpose detection model. Easily repurposed for your own use case!

## Performance Benchmarks
Comparison of models running on AWS Lambda (ARM64 architecture) using `tests/performance/test_benchmark.py`:

| Metric | YOLOv8 Nano | RF-DETR Nano | Comparison |
| :--- | :--- | :--- | :--- |
| **Input Resolution** | **640x640** | 384x384 | YOLOv8 processes ~2.8x more pixels |
| **Inference Time (Mean)** | **43.4 ms** | 118.6 ms | YOLOv8 is ~2.7x faster |
| **Memory Delta (RSS)** | **~11.7 MB** | ~39.3 MB | YOLOv8 uses ~3.3x less dynamic memory |
| **Total Memory (RSS)** | ~378 MB | ~417 MB | Both fit well within 512MB/1GB Lambda |

## Special thanks

* [davyneven](https://github.com/trainyolo/YOLOv8-aws-lambda) for their excellent AWS deployment tutorial.

* [PierreMarieCurie](https://github.com/PierreMarieCurie/rf-detr-onnx) for their RF-DETR ONNX implementation.

### To-do

[ ] Fine-tuning tutorial

[ ] Ojito integration
