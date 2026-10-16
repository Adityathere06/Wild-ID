import os
import pickle
from PIL import Image

def load_images_from_directory(directory):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []

    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return image_paths

    for filename in os.listdir(directory):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            image_paths.append(os.path.join(directory, filename))

    return sorted(image_paths)

def save_features(features_dict, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(features_dict, f)
    print(f"[OK] Features saved to {filepath}")

def load_features(filepath):
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist")
        return {}

    with open(filepath, 'rb') as f:
        features_dict = pickle.load(f)
    print(f"[OK] Features loaded from {filepath}")
    return features_dict

def parse_reference_structure(reference_dir):
    individuals = {}

    if not os.path.exists(reference_dir):
        return individuals

    for individual_name in os.listdir(reference_dir):
        individual_path = os.path.join(reference_dir, individual_name)
        if os.path.isdir(individual_path):
            images = load_images_from_directory(individual_path)
            if images:
                individuals[individual_name] = images

    return individuals

def validate_dataset(reference_dir, test_dir):
    issues = []

    if not os.path.exists(reference_dir):
        issues.append(f"Reference directory '{reference_dir}' does not exist")
    else:
        individuals = parse_reference_structure(reference_dir)
        if not individuals:
            issues.append("No individuals found in reference directory")

    if not os.path.exists(test_dir):
        issues.append(f"Test directory '{test_dir}' does not exist")

    if issues:
        return False
    return True