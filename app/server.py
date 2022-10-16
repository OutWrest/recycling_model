import sys
import torch

import uvicorn
from fastapi import FastAPI, UploadFile
from preload import ImageDataset, default_transforms, index_and_embed_images
import io
from PIL import Image

app = FastAPI()
model, data_index, embeddings, super_classes, classes, knn = index_and_embed_images()

@app.get("/")
async def index():
    return data_index

@app.post("/search")
async def create_upload_file(uploaded_img: UploadFile):
    img = await uploaded_img.read()
    img = Image.open(io.BytesIO(img))
    img = img.convert('RGB')
    img = default_transforms(img)
    img = ImageDataset.to_device({
            'image': img, 'super_class': None, 'class': None
    }, 'cuda')

    img['image'] = img['image'].unsqueeze(0)

    with torch.no_grad():
        embedding = model(img)['image_features'].cpu().numpy()

    dist, indices = knn.kneighbors(embedding)

    print(indices)
    print(super_classes[indices])
    print(classes[indices])
    print(dist)

if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=1337)