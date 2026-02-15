#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail
set -e

if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi

# Function to display help
help_function() {
    cat <<EOF
Usage: $(basename "$0") [-h] [--lib-dir LIB_DIR]

Options:
  -h, --help       Display this help message and exit
  --lib-dir        Optional path to the lib/ directory
  -f, --function   Name of the lambda function
EOF
}

# Main function of the script
main() {

    # Export model using local environment
    echo "Exporting YOLOv8n model to ONNX..."
    
    # Check if ultralytics is installed
    if ! uv run python -c "import ultralytics" &> /dev/null; then
        echo "Error: ultralytics is not installed. Please run 'pip install ultralytics' or install dev dependencies."
        exit 1
    fi

    # Run export
    uv run yolo export model=yolov8n.pt format=onnx
    
    # Move artifact
    if [[ -f "yolov8n.onnx" ]]; then
        mv yolov8n.onnx detection/models/
        echo "Export complete: detection/models/yolov8n.onnx"
        rm -f yolov8n.pt
    else
        echo "Error: Export failed, yolov8n.onnx not found."
        exit 1
    fi
}

main "$@"
