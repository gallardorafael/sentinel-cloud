import argparse

import onnx

import bentoml


def main(framework: str, model_name: str, model_path: str, batchable: bool):
    """Function that registers an ONNX model to BentoML's model store
    Args:
        framework: The framework of the model
        model_name: The name of the model
        model_path: The path to the model file
        batchable: Whether the model is batchable
    """
    if framework != "onnx":
        raise ValueError(f"Only support ONNX model, but got {framework}")

    model = onnx.load(model_path)

    bentoml.onnx.save_model(model_name, model, signatures={"run": {"batchable": batchable}})


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Registering a model to BentoML's model store")
    args.add_argument("--framework", type=str, required=True)
    args.add_argument("--model_name", type=str, required=True)
    args.add_argument("--model_path", type=str, required=True)
    args.add_argument("--batchable", action="store_true")
    args = args.parse_args()

    main(args.framework, args.model_name, args.model_path, True if args.batchable else False)
