import os
import json
import gc
import cv2
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, SiglipVisionModel, AutoConfig
from tqdm import tqdm
from collections import defaultdict

# --- CONFIGURATION ---
class Config:
    # Paths
    EXP_ROOT = "./Exp_WSAD_Perspective_20260126_092016"
    DATA_ROOT = "/mnt/HDD3/archive_for_leo/leo/data/AnimalDatasetV2Original"
    GT_JSON_PATH = "../1.PrepareData/LesionDetectionGT.json"
    
    # Files derived from EXP_ROOT
    CSV_PATH = os.path.join(EXP_ROOT, "master_all_folds_predictions.csv")
    RESULTS_DIR = os.path.join(EXP_ROOT, "legrad-results-504") 
    
    # Folders for analysis categories
    TP_DIR = os.path.join(RESULTS_DIR, "true_positive")   # Pred=1, GT=1
    FN_DIR = os.path.join(RESULTS_DIR, "false_negative")  # Pred=0, GT=1
    FP_DIR = os.path.join(RESULTS_DIR, "false_positive")  # Pred=1, GT=0

    # Hyperparameters
    MODEL_NAME = "google/siglip2-so400m-patch14-384"
    IMG_SIZE = 504 # Critical: Must match patch divisibility
    IOU_THRESH = 0.1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Normalization (Matches your training config)
    MEAN = [0.3788, 0.3788, 0.3788]
    STD = [0.2339, 0.2339, 0.2339]

# Ensure directories exist
os.makedirs(Config.TP_DIR, exist_ok=True)
os.makedirs(Config.FN_DIR, exist_ok=True)
os.makedirs(Config.FP_DIR, exist_ok=True)


# --- UTILITIES ---
def get_image_path(stem, root):
    """Robust image finder."""
    stem_str = str(stem)
    candidates = [stem_str]
    if stem_str.isdigit():
        val = int(stem_str)
        candidates.append(f"{val:03d}")
        candidates.append(f"{val:04d}")
        
    possible_exts = ['.jpeg', '.jpg', '.png', '.bmp', '.JPG', '.PNG']
    for base in candidates:
        for ext in possible_exts:
            p = os.path.join(root, f"{base}{ext}")
            if os.path.exists(p): return p
    return None


def compute_iou(boxA, boxB):
    """Computes Intersection over Union."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


# --- HELPER: ROBUST STATE DICT MAPPER ---
def remap_state_dict(source_dict, target_keys, prefix_handling="auto"):
    """
    Intelligently maps keys from source_dict to match target_keys.
    Handles 'vision_model.' prefix discrepancies.
    """
    new_dict = {}
    target_keys_set = set(target_keys)
    
    for k, v in source_dict.items():
        # 1. Exact match
        if k in target_keys_set:
            new_dict[k] = v
            continue
            
        # 2. Add 'vision_model.' prefix (Source: "embeddings..." -> Target: "vision_model.embeddings...")
        # This fixes the specific error you saw in __init__
        prefixed_k = f"vision_model.{k}"
        if prefixed_k in target_keys_set:
            new_dict[prefixed_k] = v
            continue

        # 3. Remove 'vision_model.' prefix (Source: "vision_model.embeddings..." -> Target: "embeddings...")
        if k.startswith("vision_model."):
            clean_k = k.replace("vision_model.", "")
            if clean_k in target_keys_set:
                new_dict[clean_k] = v
                continue
                
        # 4. Remove 'module.' prefix (from DataParallel)
        if k.startswith("module."):
            clean_k = k.replace("module.", "")
            # Try matching clean key again with rules 1-3
            if clean_k in target_keys_set:
                new_dict[clean_k] = v
            elif f"vision_model.{clean_k}" in target_keys_set:
                new_dict[f"vision_model.{clean_k}"] = v
                
    return new_dict


# --- MODEL DEFINITION ---
class InstanceSigLIP(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        
        print(f"Loading {model_name} with SANITIZED Eager Mode...")
        
        # 1. Load the Configuration
        config = AutoConfig.from_pretrained(model_name)
        
        # Target the Vision Config specifically
        if hasattr(config, "vision_config"):
            vision_config = config.vision_config
        else:
            vision_config = config
            
        # FORCE Eager Mode and Output Attentions
        vision_config.attn_implementation = "eager"
        vision_config.output_attentions = True
        
        # 2. Create FRESH Vision Model (Architecture Only)
        self.vision_model = SiglipVisionModel(vision_config)
        
        # 3. Load Weights from Pretrained Source (Official)
        print("Extracting weights from pretrained source...")
        temp_full_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Extract appropriate state dict
        if hasattr(temp_full_model, 'vision_model'):
            vision_state_dict = temp_full_model.vision_model.state_dict()
        else:
            vision_state_dict = temp_full_model.state_dict()
            
        # --- FIX: ROBUST LOADING FOR INIT ---
        # The error showed self.vision_model expects keys like "vision_model.embeddings..."
        # but vision_state_dict provided "embeddings...". We remap here.
        model_keys = self.vision_model.state_dict().keys()
        remapped_dict = remap_state_dict(vision_state_dict, model_keys)
        
        # Use strict=False to avoid crashing on minor unused keys, but print check
        msg = self.vision_model.load_state_dict(remapped_dict, strict=False)
        print(f"Base Vision Weights Loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
        # ------------------------------------
        
        # Cleanup temp model
        del temp_full_model
        del vision_state_dict
        gc.collect()
        torch.cuda.empty_cache()
        
        self.embed_dim = self.vision_model.config.hidden_size
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Linear(self.embed_dim, 2)

    def forward(self, x, **kwargs):
        # Explicitly Ensure output_attentions is True
        if 'output_attentions' not in kwargs:
             kwargs['output_attentions'] = True

        outputs = self.vision_model(pixel_values=x, interpolate_pos_encoding=True, **kwargs)
        emb = outputs.pooler_output
        
        # Check if attentions exist
        if kwargs.get('output_attentions', False):
            if outputs.attentions is None:
                raise ValueError("CRITICAL: Model still returned None for attentions.")
            return self.head(self.dropout(emb)), outputs.attentions
            
        return self.head(self.dropout(emb))


# --- LEGRAD INTERPRETER ---
class SigLIPLeGrad:
    def __init__(self, model, target_layer_idx=-1):
        self.model = model
        self.target_layer_idx = target_layer_idx 

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        
        # 1. Forward Pass
        logits, attentions = self.model(x, output_attentions=True)
        
        # 2. Get Target Attention Map
        target_attn = attentions[self.target_layer_idx] # (B, Heads, N, N)
        target_attn.retain_grad()
        
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1)
            
        # 3. Backward Pass
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # 4. LeGrad Formula: ReLU(Grad * Attn)
        gradients = target_attn.grad 
        activations = target_attn    
        legrad_map = (gradients * activations).clamp(min=0)
        
        # 5. Aggregate Heads & Columns
        legrad_map = legrad_map.mean(dim=1) # Average Heads -> (B, N, N)
        
        # In SigLIP (No CLS token), we want "Global Importance" of a patch.
        # We average over the Query dimension (Dim 1) to see how much other patches attended to this key.
        heatmap = legrad_map.mean(dim=1)    # Average Columns -> (B, N)
        
        # 6. Reshape & Normalize
        heatmap = heatmap.detach().cpu().numpy()
        num_patches = heatmap.shape[1]
        grid_dim = int(np.sqrt(num_patches))
        
        heatmap = heatmap.reshape(grid_dim, grid_dim)
        heatmap = cv2.resize(heatmap, (Config.IMG_SIZE, Config.IMG_SIZE))
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / (np.max(heatmap) + 1e-8)
        
        return heatmap


def get_bboxes_otsu(heatmap, orig_size, border_pixels=16):
    """Generates boxes using Otsu + Border Suppression."""
    orig_w, orig_h = orig_size
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Border Suppression
    b = border_pixels
    heatmap_norm[:b, :] = 0 
    heatmap_norm[-b:, :] = 0
    heatmap_norm[:, :b] = 0
    heatmap_norm[:, -b:] = 0
    
    thresh_val, binary = cv2.threshold(heatmap_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2] 
    
    boxes = []
    scale_x = orig_w / Config.IMG_SIZE
    scale_y = orig_h / Config.IMG_SIZE
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 20: 
            real_x1 = int(x * scale_x)
            real_y1 = int(y * scale_y)
            real_x2 = int((x + w) * scale_x)
            real_y2 = int((y + h) * scale_y)
            boxes.append([real_x1, real_y1, real_x2, real_y2])
            
    return boxes


def load_gt_annotations():
    """Parses GT JSON."""
    try:
        with open(Config.GT_JSON_PATH, 'r') as f:
            coco_data = json.load(f)
        img_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}
        img_id_to_anns = defaultdict(list)
        for ann in coco_data['annotations']:
            img_id_to_anns[ann['image_id']].append(ann['bbox'])
        gt_dict = {}
        for img_id, full_name in img_id_to_name.items():
            stem = full_name.rsplit('.', 1)[0]
            if img_id in img_id_to_anns:
                boxes = []
                for (x, y, w, h) in img_id_to_anns[img_id]:
                    boxes.append([x, y, x+w, y+h])
                gt_dict[stem] = boxes
        print(f"Loaded {len(gt_dict)} GT annotations.")
        return gt_dict
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return {}


# --- MAIN PIPELINE ---
def main():
    print("Starting LeGrad WSAD Pipeline (Fixed Load + Robust Weights)...")
    
    gt_dict = load_gt_annotations()
    df = pd.read_csv(Config.CSV_PATH)
    
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.MEAN, std=Config.STD)
    ])

    all_pred_boxes = {}

    for fold, group in df.groupby('cv_fold'):
        print(f"\nProcessing Fold {fold} ({len(group)} images)...")
        
        # Initialize Model (This now uses the fixed __init__ logic)
        model = InstanceSigLIP(Config.MODEL_NAME)
        
        w_path_only = os.path.join(Config.EXP_ROOT, f"Fold_{fold}", f"weights_only_fold_{fold}.pth")
        w_path_best = os.path.join(Config.EXP_ROOT, f"Fold_{fold}", f"best_weights_fold_{fold}.pth")
        w_path = w_path_only if os.path.exists(w_path_only) else w_path_best
        
        # --- ROBUST WEIGHT LOADING (FIXED) ---
        try:
            print(f"Loading trained weights from: {w_path}")
            checkpoint = torch.load(w_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Use the robust remapper helper
            model_keys = model.state_dict().keys()
            new_state_dict = remap_state_dict(state_dict, model_keys)

            # Load with strict=False to be robust against unused keys
            msg = model.load_state_dict(new_state_dict, strict=False)
            print(f"Weights Loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
            
            # CRITICAL CHECK: Ensure the HEAD was loaded
            head_missing = [k for k in msg.missing_keys if "head.weight" in k]
            if head_missing:
                print(f"WARNING: Classification Head weights were NOT loaded: {head_missing}")
                print("Results may be random!")
            
            model.to(Config.DEVICE)
            model.eval()
            
        except Exception as e:
            print(f"CRITICAL ERROR loading weights for Fold {fold}: {e}")
            continue
        # -----------------------------

        interpreter = SigLIPLeGrad(model, target_layer_idx=-1)
        batch_counter = 0
        
        for _, row in tqdm(group.iterrows(), total=len(group)):
            gt_val = row['ground_truth']
            pred_val = row['prediction']
            stem = str(row['filename'])

            # Determine Category
            if gt_val == 1 and pred_val == 1:
                save_dir = Config.TP_DIR
            elif gt_val == 1 and pred_val == 0:
                save_dir = Config.FN_DIR
            elif gt_val == 0 and pred_val == 1:
                save_dir = Config.FP_DIR
            else:
                continue # Skip True Negative

            img_path = get_image_path(stem, Config.DATA_ROOT)
            if not img_path: continue
            
            try:
                pil_img = Image.open(img_path).convert('RGB')
                orig_size = pil_img.size
                input_tensor = transform(pil_img).unsqueeze(0).to(Config.DEVICE)
                
                # Run LeGrad on Class 1 (Lesion)
                heatmap = interpreter(input_tensor, class_idx=1)
                
                pred_boxes = get_bboxes_otsu(heatmap, orig_size)
                all_pred_boxes[stem] = pred_boxes
                
                # Visualization
                heatmap_vis = cv2.resize(heatmap, orig_size)
                heatmap_vis = cv2.applyColorMap(np.uint8(255 * heatmap_vis), cv2.COLORMAP_JET)
                heatmap_vis = cv2.cvtColor(heatmap_vis, cv2.COLOR_BGR2RGB)
                
                overlay = cv2.addWeighted(np.array(pil_img), 0.6, heatmap_vis, 0.4, 0)
                
                color = (0, 255, 0)
                for (x1, y1, x2, y2) in pred_boxes:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
                
                fname = f"{stem}_GT{gt_val}_Pred{pred_val}.jpg"
                Image.fromarray(overlay).save(os.path.join(save_dir, fname))
                
            except Exception as e:
                print(f"Error on {stem}: {e}")
            
            finally:
                batch_counter += 1
                if batch_counter % 20 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        del model, interpreter
        gc.collect()
        torch.cuda.empty_cache()

    # Save Results
    with open(os.path.join(Config.RESULTS_DIR, "legrad_predictions.json"), 'w') as f:
        json.dump(all_pred_boxes, f)

    print("\nCalculating Metrics...")
    TP, FP, FN = 0, 0, 0
    
    for stem, gt_boxes in gt_dict.items():
        if stem not in all_pred_boxes:
            # Note: This logic assumes we processed ALL positives (Pred=1 or GT=1)
            FN += len(gt_boxes)
            continue
            
        p_boxes = all_pred_boxes[stem]
        
        # Calculate Recall
        for g_box in gt_boxes:
            detected = any(compute_iou(g_box, p) >= Config.IOU_THRESH for p in p_boxes)
            if detected: TP += 1
            else: FN += 1
                
        # Calculate Precision
        for p_box in p_boxes:
            matched = any(compute_iou(g, p_box) >= Config.IOU_THRESH for g in gt_boxes)
            if not matched: FP += 1

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    results = pd.DataFrame([{
        'IoU Threshold': Config.IOU_THRESH,
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1': round(f1, 4),
        'TP': TP, 'FP': FP, 'FN': FN
    }])

    print("\n=== FINAL DETECTION METRICS ===")
    print(results)
    results.to_csv(os.path.join(Config.RESULTS_DIR, "metrics.csv"), index=False)

if __name__ == "__main__":
    main()