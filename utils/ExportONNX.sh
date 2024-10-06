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

    # Determine the CPU architecture
    cpu_arch=$(uname -m)
    # Builds ONNX model 9 instead of 10
    local TAG="8.1.47-cpu"

    # Deploy updates to AWS
    docker run --rm -it \
        -v $(pwd):/workspace \
        -w /workspace \
        ultralytics/ultralytics:"$TAG" \
        yolo detect export model=yolov8n.pt format=onnx && \
        rm yolov8n.pt && \
        mv yolov8n.onnx detection/models
}

main "$@"
