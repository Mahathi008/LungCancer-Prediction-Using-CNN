import os
import shutil
import random
from pathlib import Path


base_dir = "raw_dataset"  
dataset_dir = "dataset"   


source_base = os.path.join(base_dir, "The IQ-OTHNCCD lung cancer dataset")


for folder in ["train", "valid", "test"]:
    for label in ["cancerous", "non-cancerous"]:
        os.makedirs(os.path.join(dataset_dir, folder, label), exist_ok=True)


def distribute_images(src_folder, label, cancerous=True):

    files = list(Path(src_folder).glob("*.jpg"))
    random.shuffle(files)  

    
    split1 = int(0.7 * len(files))  
    split2 = int(0.85 * len(files))  

    
    for i, file in enumerate(files):
        if i < split1:
            split = 'train'
        elif i < split2:
            split = 'valid'
        else:
            split = 'test'

        
        class_folder = "cancerous" if cancerous else "non-cancerous"
        dest = os.path.join(dataset_dir, split, class_folder, file.name)

        
        shutil.copy(file, dest)


distribute_images(os.path.join(source_base, "Benign cases"), "cancerous", cancerous=True)
distribute_images(os.path.join(source_base, "Malignant cases"), "cancerous", cancerous=True)


distribute_images(os.path.join(source_base, "Normal cases"), "non-cancerous", cancerous=False)

print("✅ Dataset has been split into train/valid/test folders.")
