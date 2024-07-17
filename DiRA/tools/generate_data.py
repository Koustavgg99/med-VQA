import numpy as np 
import PIL.Image as Image
import pickle
import json
import os
import pickle

# creating imgid2idx for SLAKE dataset
'''
Slake_dir = r'D:\PhD\Project\Slake\Slake1.0'
train_answers = json.load(open(os.path.join(Slake_dir, 'train.json'), encoding="utf8"))
val_answers = json.load(open(os.path.join(Slake_dir, 'validate.json'), encoding="utf8"))
test_answers = json.load(open(os.path.join(Slake_dir, 'test.json'), encoding="utf8"))
imgid2idx = {}
#print(train_answers)
for entry in train_answers:
    imgid2idx[entry['img_name']] = entry['img_id']

for entry in val_answers:
    imgid2idx[entry['img_name']] = entry['img_id']

for entry in test_answers:
    imgid2idx[entry['img_name']] = entry['img_id']

fout = open(os.path.join(Slake_dir,'imgid2idx.json'),'w')
json.dump(imgid2idx,fout,indent=1)
'''

size = 128
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
Slake_dir = r'D:\PhD\Project\Slake\Slake1.0'
img_id2idx = json.load(open(os.path.join(Slake_dir,'imgid2idx.json')))
#a = "synpic29795.jpg"
#print(img_id2idx[a])
#print(len(img_id2idx.keys()))

data_dir = r'D:\PhD\Project\Slake\Slake1.0\imgs'
img_names = list(img_id2idx.keys())

final_np = np.zeros((642, 3, size, size), dtype=np.float32)

for img_name in img_names:
    img_path = os.path.join(data_dir, img_name)
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
    #print(img_np.shape)
    final_np[img_id2idx[img_name]] = img_np

with open(os.path.join(Slake_dir,'images128x128.pkl'), 'wb') as f:
    pickle.dump(final_np, f)


#img_id2idx = json.load(open('D:\PhD\Project\VQA_RAD\imgid2idx.json'))

for key, value in img_id2idx.items():
    if value == 202:
        print(key)

data = pickle.load(open(os.path.join(Slake_dir,'images128x128.pkl'), 'rb'))
print(data.shape)
png_202 = data[202]
png_202 = (png_202 * 255).astype(np.uint8)
print(png_202.shape)
png_202 = Image.fromarray(png_202.transpose(1,2,0))
png_202.show()
