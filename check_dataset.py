import os

base_dir = "dataset"
sets = ['train', 'valid', 'test']
classes = ['cancerous', 'non-cancerous']

for set_name in sets:
    for class_name in classes:
        path = os.path.join(base_dir, set_name, class_name)
        num_images = len(os.listdir(path))
        print(f"{set_name}/{class_name}: {num_images} images")
