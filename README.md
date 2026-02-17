![Ophanim](docs/ophanim.jpg)

# $0phanim

[![Demo](https://img.shields.io/badge/Demo-Live-brightgreen?logo=github)](https://jimothyjohn.github.io/ophanim/)

A cost-effective way to detect objects in your environment. Each query completes in about a second (after bootup) and costs fractions of a penny.

**You can run a THOUSAND detections for only TWO cents!**

## Quickstart

```bash
# Format, validate, build, test, and publish 
./Quickstart -p
```

### Powered by:

* [AWS SAM](https://aws.amazon.com/serverless/sam/): Automated deployment with a single command. Accessible anyime, anywhere

* [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime): Accelerated CPU runtime. No expensive GPUs

* [RF-DETR](https://github.com/roboflow/rf-detr): Pretrained, general purpose detection/segmentation transformer. Easily repurposed for your own use case

## Special thanks

* [davyneven](https://github.com/trainyolo/YOLOv8-aws-lambda) for their excellent AWS deployment tutorial.

* [PierreMarieCurie](https://github.com/PierreMarieCurie/rf-detr-onnx) for their RF-DETR ONNX implementation.

### To-do

[ ] Fine-tuning tutorial

[ ] Ojito integration
