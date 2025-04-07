import json
import os

# Get the path to the laionc/ folder 
PACKAGE_DIR = os.path.dirname(__file__)

default_config = {
    "batch_size": 32,
    "num_workers": 4,
    "dataset_location": "./mappings/imagenet_val", 
    "index_class_path": os.path.join(PACKAGE_DIR, "mappings", "imagenet_class_index.json"),
    "super_class_path": os.path.join(PACKAGE_DIR, "mappings", "category_mapping.json"),
    "corruption_types":['glitched','mosaic','stickers','vertical_lines','geometric_shapes','luminance_checkerboard']
}

# Function to load and prepare class mappings
def load_class_mappings(index_class_path=None, super_class_path=None):
    index_class_path = index_class_path or default_config["index_class_path"]
    super_class_path = super_class_path or default_config["super_class_path"]

    # Load JSON files
    with open(index_class_path, 'r') as f:
        imagenet_class_index = json.load(f)

    with open(super_class_path, 'r') as f:
        super_class_mapping = json.load(f)
    
    # Invert category mapping and create superclass categories
    inverted_category_mapping = {
        imagenet_id: category
        for category, ids in super_class_mapping.items()
        for imagenet_id in ids
    }
    
    categories = [str(i) for i in range(1000)]
    superclass_categories = [inverted_category_mapping.get(imagenet_class_index[idx][0]) for idx in categories]
    
    return superclass_categories

