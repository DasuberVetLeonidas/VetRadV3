import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import cv2
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils as vutils
from sklearn.metrics import roc_auc_score, f1_score
import torch.nn.functional as F
from transformers import AutoModel
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm

# --- CONFIGURATION ---
# 1. Paths
DATA_ROOT = "/mnt/HDD3/archive_for_leo/leo/data/MURA"
TRAIN_CSV = os.path.join(DATA_ROOT, "train_image_csv_dataset.csv")
VALID_CSV = os.path.join(DATA_ROOT, "valid_image_csv_dataset.csv")

# 2. Resuming Configuration
RESUME_CHECKPOINT = "/mnt/HDD3/archive_for_leo/leo/models/AnimalRadV3/MURA-Pretraining/run_20260117_124022_REGULARIZED/full_model_checkpoint.pth" 

# 3. New Hyperparameters
NEW_DROPOUT = 0.5         
NEW_WEIGHT_DECAY = 0.1    
NEW_LR = 1e-5             
IMG_SIZE = 512
BATCH_SIZE = 2
GRAD_ACCUMULATION = 4
EPOCHS = 200
PATIENCE = 25
NUM_WORKERS = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 4. New Save Directory
BASE_SAVE_DIR = "/mnt/HDD3/archive_for_leo/leo/models/AnimalRadV3/MURA-Pretraining"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(BASE_SAVE_DIR, f"run_{TIMESTAMP}_RESUMED_REGULARIZED")
LOG_DIR = os.path.join(RUN_DIR, "logs")
MODEL_DIR = RUN_DIR 

# Normalization Stats (MURA Specific)
MURA_MEAN = [0.1523, 0.1523, 0.1523]
MURA_STD  = [0.1402, 0.1402, 0.1402]

ANATOMY_MAP = {
    'XR_ELBOW': 0, 'XR_FINGER': 1, 'XR_FOREARM': 2, 'XR_HAND': 3,
    'XR_HUMERUS': 4, 'XR_SHOULDER': 5, 'XR_WRIST': 6
}

# --- HELPER: Denormalize ---
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    return (tensor * std + mean).clamp(0, 1)

# --- HELPER 1: Burn Train Labels (GT Only) ---
def add_labels_to_images(images, labels, anatomies, anatomy_map_inv):
    annotated_images = []
    images_np = images.permute(0, 2, 3, 1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    anatomies_np = anatomies.cpu().numpy()
    
    for i in range(len(images_np)):
        img = images_np[i].copy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if i < len(labels_np):
            dx_text = "POS" if labels_np[i] == 1 else "NEG"
            anat_text = anatomy_map_inv.get(anatomies_np[i], "UNK").split('_')[1] 
            full_text = f"{dx_text} | {anat_text}"
            color = (0, 255, 0) if labels_np[i] == 1 else (0, 0, 255)
        else:
            full_text = "ERR"
            color = (255, 255, 0)

        cv2.putText(img, full_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        annotated_images.append(img_tensor)
        
    return torch.stack(annotated_images)

# --- HELPER 2: Burn Val Labels + Predictions ---
def add_preds_to_images(images, labels, anatomies, probs, anatomy_map_inv):
    """
    Burns GT Label AND Model Prediction onto the image.
    Color is GREEN if Correct, RED if Incorrect.
    """
    annotated_images = []
    images_np = images.permute(0, 2, 3, 1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    anatomies_np = anatomies.cpu().numpy()
    probs_np = probs.cpu().numpy()
    
    for i in range(len(images_np)):
        img = images_np[i].copy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if i < len(labels_np):
            # Ground Truth
            gt_text = "POS" if labels_np[i] == 1 else "NEG"
            
            # Prediction
            prob = probs_np[i]
            pred_class = 1 if prob > 0.5 else 0
            pred_text = "POS" if pred_class == 1 else "NEG"
            
            # Check correctness
            is_correct = (pred_class == labels_np[i])
            color = (0, 255, 0) if is_correct else (0, 0, 255) # Green vs Red
            
            # Formatting text
            anat_str = anatomy_map_inv.get(anatomies_np[i], "UNK").split('_')[1]
            # Line 1: Diagnosis
            text_line1 = f"GT:{gt_text} | PR:{pred_text}({prob:.2f})"
            # Line 2: Anatomy
            text_line2 = f"Anat: {anat_str}"

            # Draw
            cv2.putText(img, text_line1, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(img, text_line2, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
        else:
            cv2.putText(img, "ERR", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        annotated_images.append(img_tensor)
        
    return torch.stack(annotated_images)

# --- EARLY STOPPING ---
class EarlyStopping:
    def __init__(self, patience=20, verbose=True, save_dir='.'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.best_f1 = -1.0
        self.early_stop = False
        self.save_dir = save_dir
        
    def __call__(self, val_loss, val_f1, model, optimizer, epoch):
        # A. Track Best LOSS
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0 
            self.save_checkpoint(model, optimizer, epoch, val_loss, val_f1, "best_loss_model.pth")
            if self.verbose:
                print(f'\n[Saver] Val Loss improved to {val_loss:.6f}. Saved best_loss_model.pth')
        else:
            self.counter += 1
            if self.verbose:
                print(f'\n[Saver] Val Loss did not improve. Counter: {self.counter}/{self.patience}')

        # B. Track Best F1
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.save_checkpoint(model, optimizer, epoch, val_loss, val_f1, "best_f1_model.pth")
            if self.verbose:
                print(f'[Saver] Val F1 improved to {val_f1:.4f}. Saved best_f1_model.pth')

        if self.counter >= self.patience:
            self.early_stop = True
            
    def save_checkpoint(self, model, optimizer, epoch, loss, f1, filename):
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'f1': f1
        }, path)

# --- TRANSFORMS & DATASET (Unchanged) ---
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp, vp = (max_wh - w) // 2, (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return transforms.functional.pad(image, padding, 0, 'constant')

def get_transforms(split='train'):
    if split == 'train':
        return transforms.Compose([
            SquarePad(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.RandomVerticalFlip(p=0.5),    
            transforms.RandomRotation(degrees=45),   
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)), 
            transforms.ColorJitter(brightness=0.3, contrast=0.3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=MURA_MEAN, std=MURA_STD)
        ])
    else:
        return transforms.Compose([
            SquarePad(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MURA_MEAN, std=MURA_STD)
        ])

class MuraStudyDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        df = pd.read_csv(csv_path)
        
        df['Anatomy_Str'] = df['File_Path'].apply(lambda x: x.split('/')[2])
        df['Study_ID'] = df['File_Path'].apply(lambda x: '_'.join(x.split('/')[2:5]))
        
        self.studies = []
        grouped = df.groupby('Study_ID')
        for _, group in grouped:
            self.studies.append({
                'paths': group['File_Path'].tolist(),
                'label': 1.0 if 'positive' in str(group['Label'].iloc[0]) else 0.0,
                'anatomy': ANATOMY_MAP.get(group['Anatomy_Str'].iloc[0], 0)
            })
            
    def __len__(self): return len(self.studies)
    
    def __getitem__(self, idx):
        study_data = self.studies[idx]
        images = []
        for rel_path in study_data['paths']:
            full_path = os.path.join(self.root_dir, rel_path)
            try:
                img = Image.open(full_path).convert('RGB')
                if self.transform: img = self.transform(img)
                images.append(img)
            except Exception as e:
                print(f"Err: {e}")
                continue
                
        if not images: return torch.zeros(1, 3, IMG_SIZE, IMG_SIZE), torch.tensor(0.0), torch.tensor(0)
        return torch.stack(images), torch.tensor(study_data['label']), torch.tensor(study_data['anatomy'])

def collate_fn(batch):
    images_list, labels_list, anat_list, study_lengths = [], [], [], []
    for imgs, lbl, anat in batch:
        images_list.append(imgs)
        labels_list.append(lbl)
        anat_list.append(anat)
        study_lengths.append(imgs.shape[0])
    return torch.cat(images_list, dim=0), torch.stack(labels_list), torch.stack(anat_list), study_lengths

# --- MODEL (Unchanged) ---
class RadiologistSigLIP2(nn.Module):
    def __init__(self, model_name, num_anatomies=7, dropout_prob=0.3):
        super().__init__()
        full_model = AutoModel.from_pretrained(model_name)
        full_model.gradient_checkpointing_enable()
        self.encoder = full_model.vision_model
        self.embed_dim = self.encoder.config.hidden_size 
        
        self.attention_pool = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.dropout = nn.Dropout(dropout_prob) 
        self.head_dx = nn.Linear(self.embed_dim, 1)
        self.head_anat = nn.Linear(self.embed_dim, num_anatomies)
        
    def forward(self, x, study_lengths):
        outputs = self.encoder(pixel_values=x, interpolate_pos_encoding=True)
        features = outputs.pooler_output 
        split_features = torch.split(features, study_lengths)
        study_embeddings = []
        for study_feats in split_features:
            attn_weights = F.softmax(self.attention_pool(study_feats), dim=0)
            weighted_study = torch.sum(study_feats * attn_weights, dim=0)
            study_embeddings.append(weighted_study)
        study_embeddings = torch.stack(study_embeddings)
        study_embeddings = self.dropout(study_embeddings)
        return self.head_dx(study_embeddings).squeeze(1), self.head_anat(study_embeddings)

# --- MAIN ---
def main():
    print(f"--- RESUMING TRAINING (SIGLIP 2) ---")
    print(f"Resume Checkpoint: {RESUME_CHECKPOINT}")
    
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR)
    ANATOMY_MAP_INV = {v: k for k, v in ANATOMY_MAP.items()}
    
    # 1. Setup Data
    train_ds = MuraStudyDataset(TRAIN_CSV, DATA_ROOT, transform=get_transforms('train'))
    valid_ds = MuraStudyDataset(VALID_CSV, DATA_ROOT, transform=get_transforms('valid'))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    
    # 2. Setup Model & Optimizer
    model = RadiologistSigLIP2("google/siglip2-so400m-patch14-384", dropout_prob=NEW_DROPOUT).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=NEW_LR, weight_decay=NEW_WEIGHT_DECAY)
    
    # 3. Load Checkpoint
    if os.path.exists(RESUME_CHECKPOINT):
        print("Loading checkpoint...")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=DEVICE)
        
        if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
        else: model.load_state_dict(checkpoint) 
            
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = NEW_LR
                param_group['weight_decay'] = NEW_WEIGHT_DECAY
            print(f"Optimizer patched: LR={NEW_LR}, WD={NEW_WEIGHT_DECAY}")
            
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from Epoch {start_epoch}")
    else:
        raise FileNotFoundError(f"Checkpoint not found")

    criterion_dx = nn.BCEWithLogitsLoss()
    criterion_anat = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, save_dir=MODEL_DIR)
    
    global_step = start_epoch * len(train_loader)
    
    print("\nStarting Resumed Training...")
    
    for epoch in range(start_epoch, EPOCHS):
        # --- TRAIN LOOP ---
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
        
        for batch_idx, (images, labels, anatomies, lengths) in enumerate(loop):
            # Log Train Images (Once per epoch, 1st batch)
            if batch_idx == 0:
                with torch.no_grad():
                    lengths_tensor = torch.tensor(lengths).to(DEVICE)
                    expanded_labels = torch.repeat_interleave(labels.to(DEVICE), lengths_tensor)
                    expanded_anatomies = torch.repeat_interleave(anatomies.to(DEVICE), lengths_tensor)
                    vis_images = denormalize(images, MURA_MEAN, MURA_STD)
                    vis_images = add_labels_to_images(vis_images, expanded_labels, expanded_anatomies, ANATOMY_MAP_INV)
                    grid = vutils.make_grid(vis_images[:16], nrow=4, normalize=False)
                    writer.add_image('Train/Input_Examples', grid, global_step)

            images, labels, anatomies = images.to(DEVICE), labels.to(DEVICE).float(), anatomies.to(DEVICE).long()
            
            dx_logits, anat_logits = model(images, lengths)
            loss = criterion_dx(dx_logits, labels) + 0.5 * criterion_anat(anat_logits, anatomies)
            loss = loss / GRAD_ACCUMULATION
            loss.backward()
            
            if (batch_idx + 1) % GRAD_ACCUMULATION == 0:
                optimizer.step()
                optimizer.zero_grad()
                current_loss = loss.item() * GRAD_ACCUMULATION
                writer.add_scalar('Loss/Train_Step', current_loss, global_step)
                global_step += 1
                loop.set_postfix(loss=current_loss)
            
            train_loss += loss.item() * GRAD_ACCUMULATION
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION LOOP ---
        model.eval()
        val_loss = 0
        all_labels, all_preds = [], []
        
        # Randomly pick ONE batch index to log images for
        val_batches = len(valid_loader)
        random_vis_idx = random.randint(0, val_batches - 1)
        
        val_loop = tqdm(valid_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for idx, (images, labels, anatomies, lengths) in enumerate(val_loop):
                images, labels, anatomies = images.to(DEVICE), labels.to(DEVICE).float(), anatomies.to(DEVICE).long()
                dx_logits, anat_logits = model(images, lengths)
                loss = criterion_dx(dx_logits, labels) + 0.5 * criterion_anat(anat_logits, anatomies)
                val_loss += loss.item()
                
                probs = torch.sigmoid(dx_logits)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(probs.cpu().numpy())
                
                # --- LOG VALIDATION IMAGES (RANDOM BATCH) ---
                if idx == random_vis_idx:
                    # Expand attributes for per-image visualization
                    lengths_tensor = torch.tensor(lengths).to(DEVICE)
                    exp_labels = torch.repeat_interleave(labels, lengths_tensor)
                    exp_anatomies = torch.repeat_interleave(anatomies, lengths_tensor)
                    # Expand Probs (repeat study prob for all images in study)
                    exp_probs = torch.repeat_interleave(probs, lengths_tensor)
                    
                    # 1. Denormalize
                    vis_imgs = denormalize(images, MURA_MEAN, MURA_STD)
                    # 2. Burn GT + PRED
                    vis_imgs = add_preds_to_images(vis_imgs, exp_labels, exp_anatomies, exp_probs, ANATOMY_MAP_INV)
                    # 3. Log
                    grid = vutils.make_grid(vis_imgs[:16], nrow=4, normalize=False)
                    writer.add_image('Validation/Predictions', grid, epoch)

        avg_val_loss = val_loss / len(valid_loader)
        
        try: val_auc = roc_auc_score(all_labels, all_preds)
        except: val_auc = 0.5
            
        binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
        val_f1 = f1_score(all_labels, binary_preds)
        
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")
        
        writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val_Epoch', avg_val_loss, epoch)
        writer.add_scalar('Metric/Val_AUC', val_auc, epoch)
        writer.add_scalar('Metric/Val_F1', val_f1, epoch)
        
        early_stopping(avg_val_loss, val_f1, model, optimizer, epoch)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
            
    print(f"Training Done. Checkpoints saved in {MODEL_DIR}")
    writer.close()

if __name__ == "__main__":
    main()