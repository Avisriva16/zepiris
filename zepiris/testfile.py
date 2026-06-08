import os
import numpy as np
import tensorflow as tf
from PIL import Image

def softmax(x):
    """Compute softmax values to transform raw outputs into clear percentages."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def run_lite_inference(tflite_filename, image_path, apply_imagenet_norm=True):
    if not os.path.exists(tflite_filename):
        print(f"Cannot find the file '{tflite_filename}' in this folder.")
        return None

    print(f"   Running model: {tflite_filename}")

    # Load the flat standalone model binary into memory
    interpreter = tf.lite.Interpreter(model_path=tflite_filename)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load image and resize to 224x224
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))

    # Convert to numpy array and scale pixels between 0.0 and 1.0
    input_data = np.array(img, dtype=np.float32) / 255.0  

    if apply_imagenet_norm:
        # Standard normalization for Blur and NSFW models
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        input_data = (input_data - mean) / std
    else:
        # Bypassing the heavy scaling allows the anti-spoofing graph to break out of saturation 
        # Scale simply from [0, 1] to [-1, 1] range which is native to MobileNetV3 architectures
        input_data = (input_data * 2.0) - 1.0

    # Transpose to CHW shape [3, 224, 224] to match the compiled PyTorch memory structure
    input_data = np.transpose(input_data, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)  

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # If the output layer size spans broad categories, isolate the top classes
    if len(raw_output) > 2:
        raw_output = raw_output[:2]

    return softmax(raw_output)

if __name__ == "__main__":
    test_image = "face.jpg"
    
    if not os.path.exists(test_image):
        print(f"Missing file: Please place an image named '{test_image}' in this folder to run tests!")
    else:
        print(f"Evaluating test image '{test_image}' across optimized LiteRT models...\n")
        
        # 1. Evaluate Blur Model Quality (Uses ImageNet Normalization)
        blur_res = run_lite_inference("blur_model.tflite", test_image, apply_imagenet_norm=True)
        if blur_res is not None:
            print(f"   Confidences: Clear: {blur_res[0]:.2%}, Blurry: {blur_res[1]:.2%}")
            print(f"   ↳ Verdict: {'Blurry' if blur_res[1] > blur_res[0] else 'Clear'}\n")

        # 2. Evaluate NSFW Model Content Safety (Uses ImageNet Normalization)
        nsfw_res = run_lite_inference("nsfw_model.tflite", test_image, apply_imagenet_norm=True)
        if nsfw_res is not None:
            print(f"   Confidences: Safe: {nsfw_res[0]:.2%}, Unsafe: {nsfw_res[1]:.2%}")
            print(f"   ↳ Verdict: {'Unsafe/NSFW' if nsfw_res[1] > nsfw_res[0] else 'Safe'}\n")

        # 3. Evaluate Liveness Spoof Validation (Uses Native Pixel Scaling to prevent saturation)
        spoof_res = run_lite_inference("spoof_model.tflite", test_image, apply_imagenet_norm=False)
        if spoof_res is not None:
            print(f"   Confidences: Real Face: {spoof_res[0]:.2%}, Spoof/Fake: {spoof_res[1]:.2%}")
            print(f"   ↳ Verdict: {'Spoof/Fake Head' if spoof_res[1] > spoof_res[0] else 'Real Live Face'}\n")