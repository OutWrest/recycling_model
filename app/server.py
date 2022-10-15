import sys

import uvicorn
from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.post("/search")
async def create_upload_file(uploaded_img: UploadFile):
    img = await uploaded_img.read()

    pass
    

if __name__ == '__main__':
    if 'start' in sys.argv:
        uvicorn.run(app="server:app", host='0.0.0.0', port=1337, reload='reload' in sys.argv)