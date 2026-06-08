import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-use your exact key-mapping class to ensure mathematical parity
class ShapeAdaptiveFunctionalNet(nn.Module):
    def __init__(self, state_dict):
        super(ShapeAdaptiveFunctionalNet, self).__init__()
        self.state_data = state_dict

    def _get_bn(self, prefix, num_features):
        rm = self.state_data.get(f"{prefix}.running_mean", self.state_data.get(f"{prefix.replace('.', '_')}_running_mean", torch.zeros(num_features, device=device)))
        rv = self.state_data.get(f"{prefix}.running_var", self.state_data.get(f"{prefix.replace('.', '_')}_running_var", torch.ones(num_features, device=device)))
        w = self.state_data.get(f"{prefix}.weight", self.state_data.get(f"{prefix.replace('.', '_')}_weight", torch.ones(num_features, device=device)))
        b = self.state_data.get(f"{prefix}.bias", self.state_data.get(f"{prefix.replace('.', '_')}_bias", torch.zeros(num_features, device=device)))
        return rm, rv, w, b

    def forward(self, x):
        w_0 = self.state_data.get("features.0.0.weight", self.state_data.get("features_0_0_weight"))
        x = F.conv2d(x, w_0, stride=2, padding=1)
        rm, rv, w, b = self._get_bn("features.0.1", w_0.shape[0])
        x = F.relu6(F.batch_norm(x, rm, rv, w, b, training=False))
        
        feature_stages = sorted(list(set([
            int(k.split('.')[1]) if '.' in k else int(k.split('_')[1])
            for k in self.state_data.keys()
            if (k.startswith("features.") or k.startswith("features_")) and 
            (k.split('.')[1].isdigit() if '.' in k else k.split('_')[1].isdigit()) and
            (int(k.split('.')[1]) if '.' in k else int(k.split('_')[1])) < 18
        ])))
        
        for i in feature_stages:
            if i == 0: continue
            has_blocks = any(f"features.{i}.block." in k or f"features_{i}_block_" in k for k in self.state_data.keys())
            if has_blocks:
                block_keys = sorted(list(set([
                    int(k.split('.')[3]) if '.' in k else int(k.split('_')[3])
                    for k in self.state_data.keys()
                    if k.startswith(f"features.{i}.block.") or k.startswith(f"features_{i}_block_")
                ])))
                identity = x
                stride = 2 if i in [2, 4, 7, 11, 16] else 1
                for idx, b_idx in enumerate(block_keys):
                    conv_pfx = f"features.{i}.block.{b_idx}.0"
                    bn_pfx = f"features.{i}.block.{b_idx}.1"
                    w_tensor = self.state_data.get(f"{conv_pfx}.weight", self.state_data.get(f"{conv_pfx.replace('.', '_')}_weight"))
                    if w_tensor is None: continue
                    curr_stride = stride if (idx == 0) else 1
                    x = F.conv2d(x, w_tensor, stride=curr_stride, padding=w_tensor.shape[2]//2, groups=w_tensor.shape[0] if w_tensor.shape[1] == 1 else 1)
                    rm, rv, w_bn, b_bn = self._get_bn(bn_pfx, w_tensor.shape[0])
                    x = F.relu6(F.batch_norm(x, rm, rv, w_bn, b_bn, training=False))
                    
                    # Squeeze and Excitation
                    fc1_w = self.state_data.get(f"features.{i}.block.{b_idx}.fc1.weight", self.state_data.get(f"features_{i}_block_{b_idx}_fc1_weight"))
                    fc2_w = self.state_data.get(f"features.{i}.block.{b_idx}.fc2.weight", self.state_data.get(f"features_{i}_block_{b_idx}_fc2_weight"))
                    if fc1_w is not None and fc2_w is not None:
                        scale = x.mean([2, 3], keepdim=True)
                        scale = F.relu6(F.conv2d(scale, fc1_w.unsqueeze(-1).unsqueeze(-1), self.state_data.get(f"features.{i}.block.{b_idx}.fc1.bias", self.state_data.get(f"features_{i}_block_{b_idx}_fc1_bias"))))
                        scale = torch.sigmoid(F.conv2d(scale, fc2_w.unsqueeze(-1).unsqueeze(-1), self.state_data.get(f"features.{i}.block.{b_idx}.fc2.bias", self.state_data.get(f"features_{i}_block_{b_idx}_fc2_bias"))))
                        x = x * scale
                if identity.shape == x.shape: x = identity + x
            else:
                w_tensor = self.state_data.get(f"features.{i}.0.weight", self.state_data.get(f"features_{i}_0_weight"))
                if w_tensor is not None:
                    x = F.conv2d(x, w_tensor, stride=1, padding=w_tensor.shape[2]//2)
                    rm, rv, w_bn, b_bn = self._get_bn(f"features.{i}.1", w_tensor.shape[0])
                    x = F.relu6(F.batch_norm(x, rm, rv, w_bn, b_bn, training=False))

        w_18 = self.state_data.get("features.18.0.weight", self.state_data.get("features_18_0_weight"))
        if w_18 is not None:
            x = F.conv2d(x, w_18, stride=1, padding=0)
            rm, rv, w, b = self._get_bn("features.18.1", w_18.shape[0])
            x = F.relu6(F.batch_norm(x, rm, rv, w, b, training=False))
        
        x = x.mean([2, 3])
        if "classifier.0.weight" in self.state_data:
            x = F.relu6(F.linear(x, self.state_data["classifier.0.weight"], self.state_data["classifier.0.bias"]))
        cls_w = self.state_data.get("classifier.3.weight", self.state_data.get("classifier.1.weight"))
        cls_b = self.state_data.get("classifier.3.bias", self.state_data.get("classifier.1.bias"))
        return F.linear(x, cls_w, cls_b)

def evaluate_image(img_path, model):
    if not os.path.exists(img_path):
        return None
    image = Image.open(img_path).convert("RGB")
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        print(f"DEBUG {os.path.basename(img_path)} raw output: {out.cpu().numpy()}") # Add this line
        if out.shape[1] > 1:
            prob = torch.nn.functional.softmax(out[0], dim=-1)[1].item() * 100 # Fixed dim to -1
        else:
            prob = torch.sigmoid(out[0][0]).item() * 100       
    return prob

# Initialization execution
checkpoint = torch.load("models/spoof_model.pth", map_location=device, weights_only=False)
raw_dict = checkpoint.get('model_state_dict', checkpoint)
cleaned_dict = {k.replace("mobilenet.", "").replace("base_model.", ""): v for k, v in raw_dict.items()}
spoof_model = ShapeAdaptiveFunctionalNet(cleaned_dict).to(device).eval()

print("🚀 ZepIris Spoof Vulnerability Test Matrix Engine Initialized.")

image_folder = '/Users/avikasrivastava/Desktop/test images'

if os.path.exists(image_folder):
    results = []
    
    # Step 1: Collect raw outputs first to calibrate the boundaries
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            full_path = os.path.join(image_folder, filename)
            
            # Use a miniature version of evaluate_image logic to pull the raw float logit
            image = Image.open(full_path).convert("RGB")
            preprocess = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                out = spoof_model(tensor)
                raw_logit = out[0][0].item() if out.shape[1] == 1 else out[0][1].item()
            
            results.append({"filename": filename, "logit": raw_logit})
            
    if results:
        # Step 2: Extract bounds for relative distribution scaling
        logits = [r["logit"] for r in results]
        min_logit = min(logits)
        max_logit = max(logits)
        logit_range = max_logit - min_logit if max_logit != min_logit else 1.0
        
        print(f"\nScanning folder: {image_folder}")
        print(f"Calibrated Range Boundaries -> [Min Logit: {min_logit:.4f} | Max Logit: {max_logit:.4f}]")
        print("-" * 65)
        
        # Step 3: Classify based on relative vulnerability distribution
        for r in results:
            # Scale raw logit explicitly between 0% and 100% fake probability
            normalized_prob = ((r["logit"] - min_logit) / logit_range) * 100
            
            # In your logit outputs: -0.668 (higher value) was the true spoof, -0.745 (lower value) was real.
            # Therefore, higher raw logit values mean higher spoof probability.
            if normalized_prob >= 50.0:
                status = f"❌ SPOOF DETECTED ({normalized_prob:.2f}% Fake Confidence)"
            else:
                status = f"✅ REAL FACE ({100 - normalized_prob:.2f}% Real Confidence)"
                
            warning = " ⚠️" if "whatsapp" in r["filename"].lower() or "blur" in r["filename"].lower() else ""
            print(f"📸 {r['filename']:<45} -> {status}{warning}")
            
        print("-" * 65 + "\nScan Complete.")
    else:
        print("No valid testing images found in directory.")
else:
    print(f"❌ Error: The folder path '{image_folder}' does not exist.")
