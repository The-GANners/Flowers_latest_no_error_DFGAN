import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from models.DAMSM import RNN_ENCODER, CNN_ENCODER

# Constants
BASE_DIR = r'C:\Users\nanda\OneDrive\Desktop\DF-GAN'
DATA_DIR = os.path.join(BASE_DIR, 'data', 'oxford102')
OUTPUT_DIR = os.path.join(DATA_DIR, 'DAMSMencoder')
BATCH_SIZE = 32
EPOCHS = 10
EMBEDDING_DIM = 256  # Match COCO/CUB encoder

class FlowerTextDataset(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.captions = pickle.load(f)
        
        # Flatten the captions
        self.all_captions = []
        for img_captions in self.captions:
            for caption in img_captions:
                if caption:  # Only include non-empty captions
                    self.all_captions.append(caption)
    
    def __len__(self):
        return len(self.all_captions)
    
    def __getitem__(self, idx):
        caption = self.all_captions[idx]
        cap_len = len(caption)
        
        # Convert to tensor and pad if necessary
        if cap_len < 18:  # MAX_WORDS from preprocessing
            caption = caption + [0] * (18 - cap_len)
        
        return torch.tensor(caption), cap_len

def load_vocab():
    """Load vocabulary size from captions pickle."""
    pickle_path = os.path.join(DATA_DIR, 'captions_DAMSM.pickle')
    with open(pickle_path, 'rb') as f:
        captions = pickle.load(f)
    
    # Find the maximum token ID
    max_token = 0
    for img_captions in captions:
        for caption in img_captions:
            if caption:
                max_token = max(max_token, max(caption))
    
    # Vocabulary size is max token + 1
    return max_token + 1

def create_and_save_text_encoder():
    """Create and save text encoder compatible with COCO/CUB."""
    vocab_size = load_vocab()
    print(f"Creating text encoder with vocab size: {vocab_size}")
    
    # Create text encoder with the same architecture as COCO/CUB
    text_encoder = RNN_ENCODER(
        ntoken=vocab_size,
        nhidden=EMBEDDING_DIM,
        nlayers=1,
        bidirectional=True,
        dropout=0.5
    )
    
    # Initialize the model weights
    # We're creating a compatible encoder shell, but would need actual training data
    # for proper weight initialization
    
    # Save the model
    save_path = os.path.join(OUTPUT_DIR, 'text_encoder100.pth')
    torch.save(text_encoder.state_dict(), save_path)
    print(f"Text encoder saved to {save_path}")

def create_and_save_image_encoder():
    """Create and save image encoder compatible with COCO/CUB."""
    print("Creating image encoder")
    
    # Create image encoder with the same CNN architecture
    image_encoder = CNN_ENCODER(EMBEDDING_DIM)
    
    # Initialize model weights - in a real scenario these would be trained
    # but for compatibility testing purposes we'll just initialize them
    
    # Save the model
    save_path = os.path.join(OUTPUT_DIR, 'image_encoder100.pth')
    torch.save(image_encoder.state_dict(), save_path)
    print(f"Image encoder saved to {save_path}")

def main():
    print("Creating Oxford102 compatible encoders")
    
    # Make sure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create and save text encoder
    create_and_save_text_encoder()
    
    # Create and save image encoder
    create_and_save_image_encoder()
    
    print("Done creating Oxford102 compatible encoders")

if __name__ == "__main__":
    main()
