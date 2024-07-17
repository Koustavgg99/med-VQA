import numpy as np 
import PIL.Image as Image
import pickle
import json
import os
import pickle

# creating imgid2idx for CLEF dataset
size = 224
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)

CLEF_dir = r'D:\PhD\Project\VQA_Dataset'
test_imgid2idx = json.load(open(os.path.join(CLEF_dir,'test_imgid2idx.json')))

img_dir = r'D:\PhD\Project\VQA_Dataset\imgs\Test_images'
img_names = list(test_imgid2idx.keys())

final_np = np.zeros((len(img_names), 3, size, size), dtype=np.float32)

for img_name in img_names:
    img_path = os.path.join(img_dir, img_name)
    print(img_path)
    img = Image.open(img_path)
    img = img.resize((size, size), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32)
    img_np /= 255.0
    if len(img_np.shape) == 2: # if the image is not RGB
        img_np = np.stack((img_np, img_np, img_np), axis=-1)
        print(img_np.shape)
    img_np -= mean
    img_np /= std
    img_np = img_np.astype(np.float32).transpose((2, 0, 1))
    final_np[test_imgid2idx[img_name]] = img_np

with open(os.path.join(CLEF_dir,'test_images224x224.pkl'), 'wb') as f:
    pickle.dump(final_np, f)


test_img_id2idx = json.load(open(os.path.join(CLEF_dir, 'test_imgid2idx.json')))

for key, value in test_img_id2idx.items():
    if value == 202:
        print(key)

data = pickle.load(open(os.path.join(CLEF_dir,'test_images224x224.pkl'), 'rb'))
print(data.shape)
png_202 = data[202]
png_202 = (png_202 * 255).astype(np.uint8)
print(png_202.shape)
png_202 = Image.fromarray(png_202.transpose(1,2,0))
png_202.show()
