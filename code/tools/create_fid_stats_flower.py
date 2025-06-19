import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import inception_v3
import torch.nn.functional as F

def get_image_paths_from_pickle(images_dir, filenames_pickle):
    import pickle
    with open(filenames_pickle, 'rb') as f:
        filenames = pickle.load(f)
    image_paths = [os.path.join(images_dir, fname + '.jpg') for fname in filenames]
    return image_paths

def get_inception_features(image_paths, batch_size=32, device='cuda'):
    from torchvision.models import Inception_V3_Weights
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False, aux_logits=True).to(device)
    model.eval()
    resize = transforms.Resize((299, 299))
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            imgs = []
            for path in batch_paths:
                img = Image.open(path).convert('RGB')
                img = norm(to_tensor(resize(img)))
                imgs.append(img)
            imgs = torch.stack(imgs).to(device)
            # Use model.forward to get logits and features
            # We want pool3 features: model.Mixed_7c output, then AdaptiveAvgPool2d
            # But torchvision's inception_v3 returns logits by default, so we need to extract features manually
            # Use the feature extraction part only
            x = imgs
            x = model.Conv2d_1a_3x3(x)
            x = model.Conv2d_2a_3x3(x)
            x = model.Conv2d_2b_3x3(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            x = model.Conv2d_3b_1x1(x)
            x = model.Conv2d_4a_3x3(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            x = model.Mixed_5b(x)
            x = model.Mixed_5c(x)
            x = model.Mixed_5d(x)
            x = model.Mixed_6a(x)
            x = model.Mixed_6b(x)
            x = model.Mixed_6c(x)
            x = model.Mixed_6d(x)
            x = model.Mixed_6e(x)
            x = model.Mixed_7a(x)
            x = model.Mixed_7b(x)
            x = model.Mixed_7c(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            pred = x
            features.append(pred.squeeze(-1).squeeze(-1).cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features

if __name__ == "__main__":
    images_dir = r"C:\Users\nanda\OneDrive\Desktop\DF-GAN\data\flower\images"
    filenames_pickle = r"C:\Users\nanda\OneDrive\Desktop\DF-GAN\data\flower\test\filenames.pickle"
    output_npz = r"C:\Users\nanda\OneDrive\Desktop\DF-GAN\data\flower\npz\flower_val256_FIDK0.npz"
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)

    image_paths = get_image_paths_from_pickle(images_dir, filenames_pickle)
    print(f"Found {len(image_paths)} test images.")

    features = get_inception_features(image_paths, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Feature shape: {features.shape}")  # Should be (N, 2048)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    print(f"mu shape: {mu.shape}, sigma shape: {sigma.shape}")  # Should be (2048,), (2048, 2048)
    np.savez(output_npz, mu=mu, sigma=sigma)
    print(f"Saved FID stats to {output_npz}")
    image_paths = get_image_paths_from_pickle(images_dir, filenames_pickle)
    print(f"Found {len(image_paths)} test images.")

    features = get_inception_features(image_paths, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu')
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    np.savez(output_npz, mu=mu, sigma=sigma)
    print(f"Saved FID stats to {output_npz}")
