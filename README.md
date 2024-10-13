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

## Special thanks

* [davyneven](https://github.com/trainyolo/YOLOv8-aws-lambda) for their excellent AWS deployment tutorial.

### To-do

[ ] Fine-tuning tutorial

[ ] Ojito integration
