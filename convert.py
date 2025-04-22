import os
import numpy as np
from PIL import Image

def load_images_from_folder(folder, img_size=(224, 224)):
    images = []
    labels = []
    for label, class_folder in enumerate(os.listdir(folder)):
        class_path = os.path.join(folder, class_folder)
        if not os.path.isdir(class_path):
            continue
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(img_size)
                img_array = np.array(img)
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

def save_as_npz(train_folder, test_folder, output_train='train.npz', output_test='test.npz', img_size=(224, 224)):
    # Load training data
    train_images, train_labels = load_images_from_folder(train_folder, img_size)
    np.savez_compressed(output_train, images=train_images, labels=train_labels)
    print(f"Training data saved to {output_train}")

    # Load testing data
    test_images, test_labels = load_images_from_folder(test_folder, img_size)
    np.savez_compressed(output_test, images=test_images, labels=test_labels)
    print(f"Testing data saved to {output_test}")

# Specify the paths to training and testing folders
train_folder_path = "./dataset/wave/training"
test_folder_path = "./dataset/wave/testing"

# Convert and save as .npz files
save_as_npz(train_folder_path, test_folder_path)
