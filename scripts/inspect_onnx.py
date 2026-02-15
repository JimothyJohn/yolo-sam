import onnxruntime as ort
import sys


def inspect_model(model_path):
    try:
        session = ort.InferenceSession(model_path)
        print(f"Model: {model_path}")
        print("-" * 30)
        print("Inputs:")
        for i, meta in enumerate(session.get_inputs()):
            print(f"  [{i}] Name: {meta.name}, Shape: {meta.shape}, Type: {meta.type}")

        print("\nOutputs:")
        for i, meta in enumerate(session.get_outputs()):
            print(f"  [{i}] Name: {meta.name}, Shape: {meta.shape}, Type: {meta.type}")

    except Exception as e:
        print(f"Error loading model: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_onnx.py <model_path>")
    else:
        inspect_model(sys.argv[1])
