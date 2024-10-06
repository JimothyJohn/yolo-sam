![yolo](docs/yolo.jpg)

# YOLO on AWS SAM

A cost-effective way to detect objects in your environment! Each query completes in about a second and costs fractions of a penny. **You can run it a THOUSAND times for only TWO cents!**

### Powered by:

* [AWS SAM](https://aws.amazon.com/serverless/sam/): Automated deployment with a single command. Accessible anyime anywhere!

* [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime): Accelerated CPU runtime. No expensive GPUs!

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics): Pretrained, general purpose detection model. Easily repurposed for your own use case!

## Quickstart

Convenience script

```bash
# Format, Validate, Build, Test, Run
bash ./Quickstart -s
```

<details>

<summary>Export more ONNX Models</summary>

```bash
# Export nano model
bash ./utils/ExportONNX.sh
```

</details>

## Special thanks

* [davyneven](https://github.com/trainyolo/YOLOv8-aws-lambda) for his excellent AWS deployment tutorial.

### To-do

[ ] Fine-tuning tutorial
[ ] Ojito integration
