import os
import torch
import torchvision
import coremltools as ct

print("🍏 Apple Core ML Conversion Engine Started!")

def convert_pytorch_to_coreml(model_name, empty_model_skeleton):
    weights_path = f"models/{model_name}.pth"
    # FIXED: Changed output extension from .mlmodel to .mlpackage to match coremltools modern mlprogram format
    output_path = f"{model_name}.mlpackage"
    
    if not os.path.exists(weights_path):
        print(f"❌ Cannot find weights file at {weights_path}")
        return

    print(f"\n🔄 Loading weights for: {model_name}...")
    try:
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        raw_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Clean microservice state-dict prefix keys if necessary
        cleaned_dict = {k.replace("mobilenet.", "").replace("base_model.", ""): v for k, v in raw_dict.items()}
        
        empty_model_skeleton.load_state_dict(cleaned_dict, strict=False)
        empty_model_skeleton.eval()
    except Exception as e:
        print(f"❌ Error loading state dict: {e}")
        return

    # Create dummy frame input array matching target runtime dimensions
    example_input = torch.rand(1, 3, 224, 224)

    print(f"📦 Tracing computation graph for {model_name}...")
    traced_model = torch.jit.trace(empty_model_skeleton, example_input)

    print(f"✨ Converting {model_name} to Apple Core ML Format...")
    try:
        coreml_model = ct.convert(
            traced_model,
            inputs=[
                ct.ImageType(
                    name="image_input", 
                    shape=(1, 3, 224, 224),
                    scale=1.0/255.0, # Normalizes layout elements natively inside the Apple ANE Core
                    bias=[-0.485/0.229, -0.456/0.224, -0.406/0.225] # Hardcoded ImageNet channel adjustments
                )
            ]
        )
        
        coreml_model.save(output_path)
        print(f"✅ Success! Created Apple-ready layout: {output_path}")
    except Exception as e:
        print(f"❌ CoreML Conversion failed for {model_name}: {e}")

if __name__ == "__main__":
    # 1. Blur Model setup
    print("\nPreparing Blur Model...")
    blur_skeleton = torchvision.models.resnet18()
    blur_skeleton.fc = torch.nn.Linear(blur_skeleton.fc.in_features, 2)
    convert_pytorch_to_coreml("blur_model", blur_skeleton)

    # 2. NSFW Model setup
    print("\nPreparing NSFW Model...")
    nsfw_skeleton = torchvision.models.mobilenet_v2()
    nsfw_skeleton.classifier[1] = torch.nn.Linear(nsfw_skeleton.classifier[1].in_features, 2)
    convert_pytorch_to_coreml("nsfw_model", nsfw_skeleton)

    # 3. Spoof Model setup
    print("\nPreparing Spoof Model...")
    spoof_skeleton = torchvision.models.mobilenet_v3_large()
    # FIXED: Set out_features to 1 to match your [1, 1280] checkpoint array layout exactly
    spoof_skeleton.classifier[3] = torch.nn.Linear(spoof_skeleton.classifier[3].in_features, 1)
    convert_pytorch_to_coreml("spoof_model", spoof_skeleton)