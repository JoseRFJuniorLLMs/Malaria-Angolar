
import json
import os
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO
import cv2
from tqdm import tqdm

def setup_directories(base_dir):
    """Creates the YOLO directory structure."""
    dirs = ['train', 'val', 'test']
    for d in dirs:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(base_dir, d, subdir), exist_ok=True)
    return base_dir

def convert_bbox(size, box):
    """Converts (min_r, min_c, max_r, max_c) to YOLO (x, y, w, h)."""
    # box is {'minimum': {'r': r1, 'c': c1}, 'maximum': {'r': r2, 'c': c2}}
    dw = 1. / size[0]
    dh = 1. / size[1]
    
    min_r = box['minimum']['r']
    min_c = box['minimum']['c']
    max_r = box['maximum']['r']
    max_c = box['maximum']['c']
    
    x = (min_c + max_c) / 2.0
    y = (min_r + max_r) / 2.0
    w = max_c - min_c
    h = max_r - min_r
    
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def process_dataset(json_path, images_source_dir, output_base, split_name):
    """Processes a JSON file and moves images/creates labels for YOLO."""
    print(f"Processing {split_name} data from {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Target classes to detect (map all to 0: malaria_parasite)
    # Adjust this list if you want to detect other things like 'red blood cell'
    TARGET_CLASSES = ["trophozoite", "ring", "schizont", "gametocyte"]
    
    # If using formatted output directory
    images_dest_dir = os.path.join(output_base, split_name, 'images')
    labels_dest_dir = os.path.join(output_base, split_name, 'labels')
    
    processed_count = 0
    
    for item in tqdm(data):
        image_info = item['image']
        objects = item['objects']
        
        # Path processing
        # image_info['pathname'] looks like "/images/filename.png"
        filename = os.path.basename(image_info['pathname'])
        src_path = os.path.join(images_source_dir, filename)
        
        if not os.path.exists(src_path):
            # Try checking if it's just the filename in the source dir
            # Sometimes paths in json might differ slightly
            if not os.path.exists(src_path):
                print(f"Warning: Image not found: {src_path}")
                continue
                
        # Filter and convert annotations
        yolo_annotations = []
        has_parasite = False
        
        img_shape = (image_info['shape']['c'], image_info['shape']['r']) # Width, Height
        
        for obj in objects:
            category = obj['category']
            if category in TARGET_CLASSES:
                has_parasite = True
                bbox = obj['bounding_box']
                # YOLO format: class x_center y_center width height
                # We map all target classes to class 0
                box_fmt = convert_bbox(img_shape, bbox)
                yolo_annotations.append(f"0 {box_fmt[0]:.6f} {box_fmt[1]:.6f} {box_fmt[2]:.6f} {box_fmt[3]:.6f}")
        
        # Copy image and write label ONLY if we want to filter empty images?
        # Usually it's good to keep background images too to reduce False Positives.
        # But for now, let's keep all valid images.
        
        # Copy image
        shutil.copy2(src_path, os.path.join(images_dest_dir, filename))
        
        # Write label file
        label_filename = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(labels_dest_dir, label_filename), 'w') as lf:
            lf.write('\n'.join(yolo_annotations))
            
        processed_count += 1
        
    print(f"Processed {processed_count} images for {split_name}.")

def main():
    # Configuration
    # Assumes script is in src/ and data is in ../ANGOLA/archive/malaria/malaria
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_ROOT = os.path.join(BASE_DIR, 'ANGOLA', 'archive', 'malaria', 'malaria')
    IMAGES_DIR = os.path.join(DATA_ROOT, 'images')
    TRAINING_JSON = os.path.join(DATA_ROOT, 'training.json')
    TEST_JSON = os.path.join(DATA_ROOT, 'test.json')
    
    OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, 'dataset_yolo_local')
    
    # Simple check
    if not os.path.exists(IMAGES_DIR):
        print(f"Error: Images directory not found at {IMAGES_DIR}")
        return

    # 1. Prepare Dataset
    if os.path.exists(OUTPUT_DATASET_DIR):
        print(f"Removing existing dataset at {OUTPUT_DATASET_DIR}")
        shutil.rmtree(OUTPUT_DATASET_DIR)
    
    setup_directories(OUTPUT_DATASET_DIR)
    
    # Process Training Data
    # For simplicity, we'll use training.json for 'train' and test.json for 'val' 
    # to get a quick start. For a real production model, you might want to split training.json
    # into train/val and use test.json for final test.
    # Let's do a 90/10 split of training.json for Train/Val, and use Test.json for Test.
    
    print("Loading valid images from training.json...")
    with open(TRAINING_JSON, 'r') as f:
        full_train_data = json.load(f)
        
    # Shuffle and Split
    random.seed(42)
    random.shuffle(full_train_data)
    split_idx = int(len(full_train_data) * 0.9)
    train_set = full_train_data[:split_idx]
    val_set = full_train_data[split_idx:]
    
    # Helper to write split json temporarily
    def process_data_list(data_list, split_name):
        temp_json = os.path.join(OUTPUT_DATASET_DIR, f'{split_name}_temp.json')
        with open(temp_json, 'w') as f:
            json.dump(data_list, f)
        process_dataset(temp_json, IMAGES_DIR, OUTPUT_DATASET_DIR, split_name)
        os.remove(temp_json)

    process_data_list(train_set, 'train')
    process_data_list(val_set, 'val')
    
    # Process Test Data
    if os.path.exists(TEST_JSON):
        process_dataset(TEST_JSON, IMAGES_DIR, OUTPUT_DATASET_DIR, 'test')
    
    # 2. Create data.yaml
    yaml_content = {
        'path': os.path.abspath(OUTPUT_DATASET_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {0: 'malaria_parasite'}
    }
    
    yaml_path = os.path.join(OUTPUT_DATASET_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)
        
    print(f"Dataset prepared at {OUTPUT_DATASET_DIR}")
    print(f"Data YAML created at {yaml_path}")
    
    # 3. Training
    print("Starting YOLOv8 Training...")
    model = YOLO('yolov8n.pt')  # Start with nano model for speed, use 'yolov8m.pt' for better accuracy
    
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        name='malaria_yolo_local',
        patience=5,
        exist_ok=True
    )
    
    print("Training Completed.")
    print(f"Best model saved at: {results.save_dir}")

if __name__ == "__main__":
    main()
