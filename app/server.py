import io
import sys
from calendar import c

import torch
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from constants import HNAME2ID
from preload import ImageDataset, default_transforms, index_and_embed_images

origins = [
    'localhost:8080',
    'http://localhost:8080',
    'http://localhost:8080/',
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    return [{
        'super_class': s_c,
        'class': c,
        'sim': d,
        'id': HNAME2ID[c],
    } for s_c, c, d in zip(super_classes[indices].tolist()[0], classes[indices].tolist()[0], dist.tolist()[0])]

if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=1337)
