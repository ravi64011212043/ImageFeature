from fastapi import FastAPI,Request
import base64
import numpy as np
import cv2
app = FastAPI()

# @app.get('/')
# def root():
#     return {"message": "Rawiwan"}

@app.get("/api/genhog")
async def HogDescriptor(pic: Request):
        json = await pic.json()
        image = json["img"]
        data_split_img = image.split(',',1)
        img_string = data_split_img[1]

        decode_img_data = base64.b64decode(img_string)

        decode_image = cv2.imdecode(np.frombuffer(decode_img_data,np.uint8),cv2.IMREAD_GRAYSCALE)

        img_new = cv2.resize(decode_image, (128,128), cv2.INTER_AREA)
        
        win_size = img_new.shape
        cell_size = (8,8)
        block_size = (16,16)
        block_stride = (8,8)
        num_bins = 9

        hog = cv2.HOGDescriptor(win_size,block_size,block_stride,cell_size,num_bins)

        hog_descriptor = hog.compute(img_new)

        hog_descriptor_list = hog_descriptor.tolist()
       
        return {"HOG": hog_descriptor_list}
    

    
