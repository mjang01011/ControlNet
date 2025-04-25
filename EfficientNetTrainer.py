import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# === Config ===
NUM_CLASSES = 103
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Dataset ===
try:
    dataset = load_dataset("EduardoPacheco/FoodSeg103", split="train")
    print(f"✅ Dataset loaded successfully. Total samples: {len(dataset)}")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


class FoodSegHFDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.data = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            sample = self.data[idx]
            image = sample["image"]
            label = sample["label"]

            # Convert image to RGB if it's not already and apply transforms
            if not isinstance(image, torch.Tensor):
                image = image.convert("RGB")
            image = self.transform(image)

            # Convert label to tensor
            label = torch.tensor(label, dtype=torch.long)

            return image, label
        except Exception as e:
            print(f"[ERROR] Error processing sample {idx}: {e}")
            # Instead of infinite recursion, return None and handle in DataLoader
            return None


def collate_fn(batch):
    # Filter out None values (samples that caused errors)
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long)  # Return empty tensors if no valid samples

    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return images, labels


train_dataset = FoodSegHFDataset(dataset, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4) # Added num_workers for faster loading


# === Model ===
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

# === Optimizer and Loss ===
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# === Training Loop ===
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        # Check if the batch is not empty
        if images.numel() == 0:
            print("⚠️ Warning: Skipping empty batch.")
            continue

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"✅ Epoch {epoch+1}: Loss = {running_loss / len(train_loader):.4f}")

# === Save Model ===
torch.save(model.state_dict(), "ingredient_classifier_foodseg103.pth")
print("✅ Model saved to ingredient_classifier_foodseg103.pth")