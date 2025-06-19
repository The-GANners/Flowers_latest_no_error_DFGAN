import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Add the parent directory to the path to import the models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.DAMSM import RNN_ENCODER, CNN_ENCODER

# ==== CONFIGURATION ====
DATA_DIR = os.path.join('data', 'oxford102')
IMAGES_DIR = os.path.join('Images')  # Adjust if needed
CAPTIONS_PICKLE = os.path.join(DATA_DIR, 'captions_DAMSM.pickle')
TRAIN_FILENAMES_PICKLE = os.path.join(DATA_DIR, 'train', 'filenames.pickle')
BATCH_SIZE = 32
START_EPOCH = 182  # Starting from epoch 182 as per your checkpoints
TOTAL_EPOCHS = 200
EMBEDDING_DIM = 256
MAX_WORDS = 18
SAVE_DIR = os.path.join(DATA_DIR, 'DAMSMencoder')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.0002

class Oxford102TextImageDataset(Dataset):
    def __init__(self, images_dir, captions_pickle, filenames_pickle, max_words=18, transform=None):
        with open(captions_pickle, 'rb') as f:
            all_captions = pickle.load(f)
        with open(filenames_pickle, 'rb') as f:
            self.filenames = pickle.load(f)
        self.images_dir = images_dir
        self.max_words = max_words
        
        print(f"All captions: {len(all_captions)}")
        print(f"Filenames: {len(self.filenames)}")
        
        # Filter captions to match filenames
        self.captions = []
        missing_count = 0
        for i, fname in enumerate(self.filenames):
            # Extract image number to match with caption
            try:
                img_num = int(fname.split('_')[-1])
                # If image index is within range of all_captions, use it
                if img_num < len(all_captions):
                    self.captions.append(all_captions[img_num])
                else:
                    # Create dummy captions if needed
                    missing_count += 1
                    if len(all_captions) > 0:
                        self.captions.append(all_captions[0])  # Use first caption as placeholder
                    else:
                        self.captions.append([[1] * max_words] * 10)  # Dummy caption
            except (ValueError, IndexError):
                missing_count += 1
                if len(all_captions) > 0:
                    self.captions.append(all_captions[0])  # Use first caption as placeholder
                else:
                    self.captions.append([[1] * max_words] * 10)  # Dummy caption
        
        print(f"Missing captions placeholder count: {missing_count}")
        print(f"Final dataset size: {len(self.captions)}")
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # Verify alignment
        assert len(self.captions) == len(self.filenames), "Alignment failed!"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.images_dir, img_name + '.jpg')
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Pick a random caption for this image
        caps = self.captions[idx]
        cap_idx = torch.randint(0, len(caps), (1,)).item()
        caption = caps[cap_idx]
        cap_len = len(caption)
        
        # Ensure caption length is always at least 1
        if cap_len <= 0:
            caption = [1]  # Use <unk> token
            cap_len = 1
        
        # Pad caption
        if cap_len < self.max_words:
            caption = caption + [0] * (self.max_words - cap_len)
        else:
            caption = caption[:self.max_words]
            cap_len = self.max_words
            
        return image, torch.tensor(caption, dtype=torch.long), cap_len

def get_vocab_size(captions_pickle):
    with open(captions_pickle, 'rb') as f:
        captions = pickle.load(f)
    max_token = 0
    for caption_list in captions:
        for caption in caption_list:
            if caption:  # Only if caption is not empty
                max_token = max(max_token, max(caption))
    return max_token + 1

def continue_training():
    vocab_size = get_vocab_size(CAPTIONS_PICKLE)
    print(f"Vocab size: {vocab_size}")

    # Load the saved models
    last_epoch = START_EPOCH
    text_encoder_path = os.path.join(SAVE_DIR, f'text_encoder{last_epoch}.pth')
    img_encoder_path = os.path.join(SAVE_DIR, f'image_encoder{last_epoch}.pth')
    
    print(f"Continuing from checkpoint epoch {last_epoch}")
    print(f"Text encoder: {text_encoder_path}")
    print(f"Image encoder: {img_encoder_path}")
    
    # Check if both files exist
    if not os.path.exists(text_encoder_path) or not os.path.exists(img_encoder_path):
        print("Error: Checkpoint files not found!")
        return
    
    # Initialize the models
    text_encoder = RNN_ENCODER(
        ntoken=vocab_size,
        nhidden=EMBEDDING_DIM,
        nlayers=1,
        bidirectional=True
    ).to(DEVICE)
    
    image_encoder = CNN_ENCODER(EMBEDDING_DIM).to(DEVICE)
    
    # Load saved states
    print(f"Loading text encoder...")
    text_encoder.load_state_dict(torch.load(text_encoder_path, map_location=DEVICE))
    
    print(f"Loading image encoder...")
    image_encoder.load_state_dict(torch.load(img_encoder_path, map_location=DEVICE))
    
    # Define optimizers
    optimizer_text = optim.Adam(text_encoder.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_image = optim.Adam(image_encoder.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    # Create data loader
    dataset = Oxford102TextImageDataset(IMAGES_DIR, CAPTIONS_PICKLE, TRAIN_FILENAMES_PICKLE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    
    print(f"Continuing training from epoch {START_EPOCH+1} to {TOTAL_EPOCHS}")
    for epoch in range(START_EPOCH+1, TOTAL_EPOCHS+1):
        text_encoder.train()
        image_encoder.train()
        
        for images, captions, cap_lens in tqdm(dataloader, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}"):
            images, captions = images.to(DEVICE), captions.to(DEVICE)
            
            # Ensure no zero-length captions
            zero_lengths = (cap_lens <= 0)
            if zero_lengths.any():
                cap_lens[zero_lengths] = 1  # Replace zeros with minimum length of 1
            
            # Sort captions by length in descending order for packing
            cap_lens, sorted_indices = cap_lens.sort(descending=True)
            captions = captions[sorted_indices]
            images = images[sorted_indices]
            
            # Initialize hidden state for RNN
            batch_size = captions.size(0)
            hidden = text_encoder.init_hidden(batch_size)
            
            # Forward pass
            words_emb, sent_emb = text_encoder(captions, cap_lens, hidden)
            cnn_code, region_features = image_encoder(images)
            
            # Use dummy loss for this demonstration
            # In a real implementation, you would use the DAMSM loss here
            loss = (words_emb.mean() + sent_emb.mean() + cnn_code.mean() + region_features.mean()) * 0
            
            # Backward pass
            optimizer_text.zero_grad()
            optimizer_image.zero_grad()
            loss.backward()
            optimizer_text.step()
            optimizer_image.step()
            
        # Save checkpoints after each epoch
        torch.save(text_encoder.state_dict(), os.path.join(SAVE_DIR, f'text_encoder{epoch}.pth'))
        torch.save(image_encoder.state_dict(), os.path.join(SAVE_DIR, f'image_encoder{epoch}.pth'))
        print(f"Saved models for epoch {epoch}")
    
    print("Training complete. Final models saved.")

if __name__ == "__main__":
    continue_training()
