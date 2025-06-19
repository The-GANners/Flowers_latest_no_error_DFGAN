import os
import pickle
import glob
import re
from collections import defaultdict
from tqdm import tqdm
import nltk
from nltk.tokenize import RegexpTokenizer

# Base directories
BASE_DIR = r'C:\Users\nanda\OneDrive\Desktop\DF-GAN'
DATA_DIR = os.path.join(BASE_DIR, 'data', 'oxford102')
IMAGES_DIR = os.path.join(BASE_DIR, 'Images')
TEXT_DIR = os.path.join(BASE_DIR, 'text_c10')

# Ensure output directories exist
os.makedirs(os.path.join(DATA_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'test'), exist_ok=True)

# Parameters
MAX_WORDS = 18

def load_class_splits():
    """Load class splits from text files."""
    # Load test classes - these are only for testing
    test_classes = set()
    with open(os.path.join(BASE_DIR, 'testclasses.txt'), 'r') as f:
        for line in f:
            test_classes.add(line.strip())
    
    # All other classes (from trainclasses.txt and valclasses.txt) are for training
    train_classes = set()
    
    # Load train classes
    with open(os.path.join(BASE_DIR, 'trainclasses.txt'), 'r') as f:
        for line in f:
            train_classes.add(line.strip())
    
    # Load val classes
    with open(os.path.join(BASE_DIR, 'valclasses.txt'), 'r') as f:
        for line in f:
            train_classes.add(line.strip())
    
    print(f"Test classes: {len(test_classes)}")
    print(f"Train classes: {len(train_classes)}")
    
    # Check for overlap or missing classes
    intersection = test_classes.intersection(train_classes)
    if intersection:
        print(f"Warning: {len(intersection)} classes appear in both train and test sets")
    
    all_classes = set(f"class_{i:05d}" for i in range(1, 103))
    missing_classes = all_classes - (train_classes.union(test_classes))
    if missing_classes:
        print(f"Warning: {len(missing_classes)} classes are missing from both train and test sets")
    
    return train_classes, test_classes

def identify_image_class(image_path):
    """Identify class from image file or metadata."""
    # You might need to adjust this logic based on your actual dataset structure
    basename = os.path.basename(image_path)
    filename = os.path.splitext(basename)[0]
    
    # Method 1: Extract class from file structure
    # If your files are organized as class_XXXXX/image_YYYYY.jpg
    
    # Method 2: Extract class from filename or metadata file
    # If you have a metadata file mapping images to classes
    
    # Method 3: Determine class by image number (assuming 80 images per class)
    # This is an estimate - adjust based on your actual dataset
    try:
        image_num = int(filename.split('_')[-1])
        class_num = ((image_num - 1) // 80) + 1
        return f"class_{class_num:05d}"
    except (ValueError, IndexError):
        print(f"Warning: Could not determine class for {basename}")
        return None

def get_class_to_images_mapping():
    """Create a mapping from class names to lists of image files."""
    class_to_images = defaultdict(list)
    
    # Get all images
    all_image_files = glob.glob(os.path.join(IMAGES_DIR, '*.jpg'))
    print(f"Found {len(all_image_files)} image files")
    
    # Map each image to its class
    for img_path in tqdm(all_image_files, desc="Mapping images to classes"):
        class_id = identify_image_class(img_path)
        if class_id:
            basename = os.path.basename(img_path)
            filename = os.path.splitext(basename)[0]
            class_to_images[class_id].append(filename)
    
    return class_to_images

def load_and_encode_captions(image_filenames):
    """Load and encode captions for all provided image filenames."""
    # Make sure NLTK resources are available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Setup tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    
    # Build vocabulary
    word_counts = defaultdict(int)
    image_to_raw_captions = {}
    
    # First pass: collect all captions and build vocabulary
    print("Loading captions and building vocabulary...")
    for img_name in tqdm(image_filenames):
        # Find caption file - adjust path construction based on your dataset
        caption_files = []
        for class_dir in glob.glob(os.path.join(TEXT_DIR, 'class_*')):
            caption_file = os.path.join(class_dir, f"{img_name}.txt")
            if os.path.exists(caption_file):
                caption_files.append(caption_file)
        
        if not caption_files:
            # Try alternative locations if needed
            continue
            
        # Use the first caption file found
        caption_file = caption_files[0]
        
        try:
            with open(caption_file, 'r', encoding='utf-8') as f:
                captions = [line.strip() for line in f if line.strip()]
                
            if len(captions) == 0:
                continue
                
            # Clean captions and update word counts
            cleaned_captions = []
            for cap in captions:
                cap = cap.lower().strip()
                cap = re.sub(r"[^a-z0-9]+", " ", cap)
                tokens = tokenizer.tokenize(cap)
                for token in tokens:
                    word_counts[token] += 1
                cleaned_captions.append(cap)
                
            image_to_raw_captions[img_name] = cleaned_captions
        except Exception as e:
            print(f"Error loading captions for {img_name}: {e}")
    
    # Build vocabulary from word counts (only keep words appearing more than once)
    words = [word for word, count in word_counts.items() if count > 1]
    # Add special tokens
    words = ['<pad>', '<unk>'] + words
    word_to_idx = {word: idx for idx, word in enumerate(words)}
    
    print(f"Vocabulary size: {len(word_to_idx)}")
    
    # Second pass: encode all captions using the vocabulary
    image_to_encoded_captions = {}
    
    print("Encoding captions...")
    for img_name, captions in tqdm(image_to_raw_captions.items()):
        encoded_caps = []
        for cap in captions:
            tokens = tokenizer.tokenize(cap)
            if len(tokens) > MAX_WORDS:
                tokens = tokens[:MAX_WORDS]
            indices = [word_to_idx.get(token, 1) for token in tokens]  # 1 is <unk>
            encoded_caps.append(indices)
        
        # Ensure we have exactly 10 captions per image
        while len(encoded_caps) < 10:
            # Duplicate existing captions or add empty ones
            if encoded_caps:
                encoded_caps.append(encoded_caps[0])  # Repeat first caption
            else:
                encoded_caps.append([1])  # Add placeholder with <unk>
                
        image_to_encoded_captions[img_name] = encoded_caps[:10]  # Keep only 10 captions
    
    return image_to_encoded_captions, word_to_idx

def main():
    print("Creating Oxford102 dataset files from scratch")
    
    # Load class splits
    train_classes, test_classes = load_class_splits()
    
    # Get mapping from classes to images
    class_to_images = get_class_to_images_mapping()
    
    # Create train and test filenames lists
    train_filenames = []
    test_filenames = []
    
    all_classes = set()
    processed_images = 0
    
    for class_id, filenames in class_to_images.items():
        all_classes.add(class_id)
        processed_images += len(filenames)
        
        if class_id in test_classes:
            test_filenames.extend(filenames)
        else:
            # All non-test classes are considered training
            train_filenames.extend(filenames)
    
    print(f"Total identified classes: {len(all_classes)}")
    print(f"Total processed images: {processed_images}")
    print(f"Train filenames: {len(train_filenames)}")
    print(f"Test filenames: {len(test_filenames)}")
    
    # Get all image filenames for caption processing
    all_filenames = train_filenames + test_filenames
    
    # Load and encode captions
    image_to_encoded_captions, word_to_idx = load_and_encode_captions(all_filenames)
    
    # Create caption lists in the format expected by DF-GAN
    train_caption_list = []
    for filename in train_filenames:
        if filename in image_to_encoded_captions:
            train_caption_list.append(image_to_encoded_captions[filename])
        else:
            print(f"Warning: No captions for {filename}, using placeholder")
            # Add placeholder captions (10 captions with just <unk>)
            train_caption_list.append([[1]] * 10)
    
    test_caption_list = []
    for filename in test_filenames:
        if filename in image_to_encoded_captions:
            test_caption_list.append(image_to_encoded_captions[filename])
        else:
            print(f"Warning: No captions for {filename}, using placeholder")
            # Add placeholder captions
            test_caption_list.append([[1]] * 10)
    
    # Combine all captions (train first, then test) for DAMSM compatibility
    all_captions = train_caption_list + test_caption_list
    
    print(f"Number of caption entries: {len(all_captions)}")
    print(f"Number of images: {len(all_filenames)}")
    
    # Verify alignment
    if len(all_captions) != len(all_filenames):
        print("Warning: Caption count does not match image count!")
    
    # Save the files
    with open(os.path.join(DATA_DIR, 'train', 'filenames.pickle'), 'wb') as f:
        pickle.dump(train_filenames, f)
    print(f"Saved {len(train_filenames)} train filenames")
    
    with open(os.path.join(DATA_DIR, 'test', 'filenames.pickle'), 'wb') as f:
        pickle.dump(test_filenames, f)
    print(f"Saved {len(test_filenames)} test filenames")
    
    with open(os.path.join(DATA_DIR, 'captions_DAMSM.pickle'), 'wb') as f:
        pickle.dump(all_captions, f)
    print(f"Saved {len(all_captions)} caption lists to captions_DAMSM.pickle")
    
    print("All Oxford102 dataset files created successfully!")

if __name__ == "__main__":
    main()
