import os, shutil

base_train = "dataset/Tomato/train"
base_val = "dataset/Tomato/val"

dest_train_healthy  = "dataset/train/Healthy"
dest_train_diseased = "dataset/train/Diseased"
dest_val_healthy    = "dataset/valid/Healthy"
dest_val_diseased   = "dataset/valid/Diseased"

os.makedirs(dest_train_healthy, exist_ok=True)
os.makedirs(dest_train_diseased, exist_ok=True)
os.makedirs(dest_val_healthy, exist_ok=True)
os.makedirs(dest_val_diseased, exist_ok=True)

def move_images(src_folder, dest_folder):
    for file in os.listdir(src_folder):
        shutil.copy(os.path.join(src_folder, file), dest_folder)

# Train folders
move_images(os.path.join(base_train, "Tomato___Healthy"), dest_train_healthy)
move_images(os.path.join(base_train, "Tomato___Early_blight"), dest_train_diseased)
move_images(os.path.join(base_train, "Tomato___Late_blight"), dest_train_diseased)

# Validation folders
move_images(os.path.join(base_val, "Tomato___Healthy"), dest_val_healthy)
move_images(os.path.join(base_val, "Tomato___Early_blight"), dest_val_diseased)
move_images(os.path.join(base_val, "Tomato___Late_blight"), dest_val_diseased)

print("✅ Dataset organized into Healthy & Diseased!")
