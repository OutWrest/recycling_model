import os
import pickle
from collections import defaultdict
from glob import glob

import numpy as np
import open_clip
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

default_transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711])
])

class ImageDataset(Dataset):
    def __init__(self, imgs, transform=default_transforms):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def to_device(batch, device):
        if isinstance(batch, dict):
            batch['image'] = batch['image'].to(device)
        else:
            for i in range(len(batch)):
                batch[i]['image'] = batch[i]['image'].to(device)
        
        return batch

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        _, super_class_, class_, _ = self.imgs[idx].split('/')
        
        return {
            'image': img,
            'super_class': super_class_,
            'class': class_,
        }

class CLIPModel_S1(torch.nn.Module):
    def __init__(self, clip_model: str, pretrained: str, device: str = 'cuda'):
        super().__init__()
        self.backbone, _, _ = open_clip.create_model_and_transforms(clip_model, pretrained, device=device)

    def forward(self, x):
        return {
            'image_features': self.backbone.encode_image(x['image']),
            'super_class': x['super_class'],
            'class': x['class'],
        }

class CLIPModel_S2(torch.nn.Module):
    def __init__(self, clip_model: str, pretrained: str, pca: str, device: str = 'cuda'):
        super().__init__()
        self.backbone, _, _ = open_clip.load(clip_model, pretrained, device=device)
        self.pca = pickle.load(open(pca, 'rb'))

    def forward(self, x):
        embed = self.backbone.encode_image(x)
        embed = self.pca.transform(embed.cpu().numpy())
        return {
            'image_features': torch.from_numpy(embed).to(x.device),
            'super_class': x['super_class'],
            'class': x['class'],
        }

def index_and_embed_images() -> nn.Module:
    model = CLIPModel_S1('ViT-H-14', 'laion2b_s32b_b79k').to('cuda')
    model.eval()

    dataset = ImageDataset(glob('data/*/*/*'))
    dataloader = DataLoader(dataset, batch_size=16, num_workers=8, shuffle=False)
    
    image_features = []
    super_classes = []
    classes = []

    for batch in tqdm(dataloader):
        batch = ImageDataset.to_device(batch, 'cuda')
        with torch.no_grad():
            batch = model(batch)
        image_features.extend(batch['image_features'].cpu().numpy())
        super_classes.extend(batch['super_class'])
        classes.extend(batch['class'])

    image_features = np.array(image_features)
    super_classes = np.array(super_classes)
    classes = np.array(classes)

    for i in range(len(image_features)):
        os.makedirs(f'cache/{super_classes[i]}/{classes[i]}', exist_ok=True)
        # np.save(f'cache/{super_classes[i]}/{classes[i]}/{i}.npy', image_features[i])

        img = Image.open(dataset.imgs[i])
        img = img.convert('RGB')
        img.save(f'cache/{super_classes[i]}/{classes[i]}/{i}.jpg')

    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(image_features)

    data_index = defaultdict(list)
    for d in glob('data/*/*'):
        s_c, c = d.split('/')[1:]

        if 'jpg' in c:
            data_index[s_c] = None
            continue

        data_index[s_c].append(c)

        

    return model, data_index, image_features, super_classes, classes, knn

if __name__ == '__main__':
    index_and_embed_images()
