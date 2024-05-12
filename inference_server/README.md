# SENTINEL Cloud: Inference Server

## How to run the server
You can start the BentoML inference server by running:

    bentoml serve service:FaceDetection

## How to register new models in BentoML
There is a utility script in the utils folder of the inference server, you just need to run the script
over your desired ONNX model:

    python3 utils/register_onnx.py --framework onnx --model_name yolov6s_face --model_path weights/yolov6s_face.onnx --batchable

And then you can verify that the model was registered sucessfully:

    bentoml models list

## List of models on the Inference Server

- yolov6s_face: YOLOv6 architecture, pre-trained and fine-tuned to solve a face detection task, and converted to ONNX according to the [YOLOv6 repository](https://github.com/meituan/YOLOv6/tree/yolov6-face)