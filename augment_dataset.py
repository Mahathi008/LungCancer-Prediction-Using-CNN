import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

 
base_dir = "dataset/train"
output_dir = "dataset/train_augmented"

 
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'cancerous'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'non-cancerous'), exist_ok=True)

 
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

 
def augment_images(class_name):
    class_path = os.path.join(base_dir, class_name)
    output_class_path = os.path.join(output_dir, class_name)
    
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        try:
            img = load_img(image_path) 
            x = img_to_array(img)  
            x = x.reshape((1,) + x.shape)  

            i = 0
            
            for batch in datagen.flow(x, batch_size=1, save_to_dir=output_class_path, save_prefix='aug_', save_format='jpeg'):
                i += 1
                if i >= 30:  
                    break
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            continue  


augment_images('cancerous')  
augment_images('non-cancerous')  

print("✅ Dataset augmentation complete.")
