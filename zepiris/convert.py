import os
import torch
import torchvision
import litert_torch  # Updated from ai_edge_torch

def convert_pytorch_to_tflite(model_name, empty_model_skeleton):
    print(f"\n Starting Native LiteRT Conversion for: {model_name}...")
    
    weights_path = os.path.join("models", f"{model_name}.pth")
    tflite_output_path = f"{model_name}.tflite"
    
    if not os.path.exists(weights_path):
        print(f" Error: Weights file '{weights_path}' not found.")
        return

    # 1. Safely bind the heavy weights to the structure template
    try:
        empty_model_skeleton.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=False))
    except Exception as e:
        empty_model_skeleton.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=False), strict=False)
    
    empty_model_skeleton.eval()

    # 2. Compile directly from PyTorch to highly optimized mobile TFLite format
    try:
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Use LiteRT's native conversion call function
        litert_model = litert_torch.convert(empty_model_skeleton, (dummy_input,))
        
        # 3. Export the clean flat asset binary file
        litert_model.export(tflite_output_path)
        print(f"   Success! Created: {tflite_output_path}")
    except Exception as e:
        print(f"Core compiler failed on {model_name}: {str(e)}")

if __name__ == "__main__":
    print("Firing Google LiteRT PyTorch Conversion Backend Engine...")
    
    # 1. Blur Model (Standard ResNet18 structure)
    convert_pytorch_to_tflite("blur_model", torchvision.models.resnet18())
    
    # 2. NSFW Model (Custom layer to output 2 classes instead of 1000)
    nsfw_skeleton = torchvision.models.mobilenet_v2()
    nsfw_skeleton.classifier[1] = torch.nn.Linear(in_features=1280, out_features=2)
    convert_pytorch_to_tflite("nsfw_model", nsfw_skeleton)
    
    # 3. Spoof Model (Custom layer to output 2 classes instead of 1000)
    spoof_skeleton = torchvision.models.mobilenet_v3_large()
    spoof_skeleton.classifier[3] = torch.nn.Linear(in_features=1280, out_features=2)
    convert_pytorch_to_tflite("spoof_model", spoof_skeleton)
    
    print("\n Process finished! Check sidebar for the flat .tflite files.")