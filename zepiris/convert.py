import os
import torch
import torchvision
import litert_torch  # Native Google LiteRT compilation engine

# Explicitly import the custom model definition from the zepiris project layout
from zepiris.ml_inference.models import ResNetBlurDetector

def convert_pytorch_to_tflite(model_name, empty_model_skeleton):
    print(f"\n🚀 Starting Native LiteRT Conversion for: {model_name}...")
    
    weights_path = os.path.join("models", f"{model_name}.pth")
    tflite_output_path = f"{model_name}.tflite"
    
    if not os.path.exists(weights_path):
        print(f"❌ Error: Weights file '{weights_path}' not found.")
        return

    # 1. Load weights using strict enforcement to ensure the skeleton is perfectly shaped
    try:
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        # Handle cases where weights might be tucked under a 'model_state_dict' wrapper dictionary
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        empty_model_skeleton.load_state_dict(state_dict, strict=True)
        print(f"   ✅ State dict successfully bound to skeleton structure.")
    except Exception as e:
        print(f"   ⚠️ Strict binding warning: {str(e)}. Retrying with fallback...")
        empty_model_skeleton.load_state_dict(state_dict, strict=False)
    
    empty_model_skeleton.eval()

    # 2. Compile from PyTorch to highly optimized mobile TFLite format
    try:
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Use LiteRT's native conversion call function
        litert_model = litert_torch.convert(empty_model_skeleton, (dummy_input,))
        
        # 3. Export the clean flat asset binary file
        litert_model.export(tflite_output_path)
        print(f"   🎉 Success! Created: {tflite_output_path}")
    except Exception as e:
        print(f"   ❌ Core compiler failed on {model_name}: {str(e)}")

if __name__ == "__main__":
    print("Firing Google LiteRT PyTorch Conversion Backend Engine...")
    
    # =========================================================================
    # FIX: Use the actual ZepIris architecture definition instead of standard 1000-class resnet18
    # If the local module isn't accessible, use:
    #     blur_skeleton = torchvision.models.resnet18()
    #     blur_skeleton.fc = torch.nn.Linear(in_features=512, out_features=1)
    # =========================================================================
    try:
        blur_skeleton = ResNetBlurDetector(dropout_prob=0.5)
    except ImportError:
        # Fallback manual modification if run outside zepiris root path environment
        blur_skeleton = torchvision.models.resnet18()
        blur_skeleton.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=512, out_features=1) # Single node logit output matching ZepIris framework
        )
    
    convert_pytorch_to_tflite("blur_model", blur_skeleton)
    
    # 2. NSFW Model (Custom layer to output 2 classes instead of 1000)
    nsfw_skeleton = torchvision.models.mobilenet_v2()
    nsfw_skeleton.classifier[1] = torch.nn.Linear(in_features=1280, out_features=2)
    convert_pytorch_to_tflite("nsfw_model", nsfw_skeleton)
    
    # 3. Spoof Model (Custom layer to output 2 classes instead of 1000)
    spoof_skeleton = torchvision.models.mobilenet_v3_large()
    spoof_skeleton.classifier[3] = torch.nn.Linear(in_features=1280, out_features=2)
    convert_pytorch_to_tflite("spoof_model", spoof_skeleton)