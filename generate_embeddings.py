import torch
import math
import os
import random
import time
import json
import torch.nn.functional as F
from models.blip_retrieval import blip_retrieval
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import numpy as np

mode = 'text'

def load_model():
    model = blip_retrieval(pretrained='model_large_retrieval_coco.pth', image_size=384, vit='large', vit_grad_ckpt=True,
                        vit_ckpt_layer=12, queue_size=57600, negative_all_rank=True)
    device = torch.device('cuda')
    model.to(device)
    model.eval()

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_test = transforms.Compose([
            transforms.Resize((384, 384),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
            ])
    
    return model, transform_test

def embed_image_batch(model, transform_test, image_ids):
    device = torch.device('cuda')
    with torch.no_grad():
        image_list = []
        used_image_ids = []
        for i in range(len(image_ids)):
            image_id = image_ids[i]
            image_path = '/cs/labs/oabend/uriber/datasets/COCO/val2014/COCO_val2014_' + str(image_id).zfill(12) + '.jpg'
            image = Image.open(image_path)
            if len(np.array(image).shape) == 2:
                continue
            image = transform_test(image)
            image = image.unsqueeze(dim=0)
            image = image.to(device)
            image_list.append(image)
            used_image_ids.append(image_id)
        
        images = torch.cat(image_list)
        image_feat = model.visual_encoder(images)
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)
    return image_embed, used_image_ids

def embed_text_batch(model, transform_test, sentences):
    device = torch.device('cuda')
    with torch.no_grad():
        text_input = model.tokenizer(sentences, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        
    return text_embed

with open('../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
    coco_data = json.load(fp)

coco_data = [x for x in coco_data['images'] if x['split'] == 'test']

print('Loading model...', flush=True)
model, transform_test = load_model()

batch_start = 0
batch_size = 100
embed_list = []
all_ids = []

t = time.time()
while batch_start < len(coco_data):
    print('Starting sample ' + str(batch_start) + ' out of ' + str(len(coco_data)) + ', time from prev ' + str(time.time()-t), flush=True)
    t = time.time()

    if mode == 'text':
        batch = []
        caption_ids = []
        while len(batch) < batch_size:
            batch += [x['raw'] for x in coco_data[batch_start]['sentences']]
            caption_ids += [x['sentid'] for x in coco_data[batch_start]['sentences']]
            batch_start = batch_start + 1
        text_embed = embed_text_batch(model, transform_test, batch)
        embed_list.append(text_embed)
        all_ids += caption_ids
    elif mode == 'image':
        batch = [coco_data[i]['cocoid'] for i in range(batch_start, batch_end)]
        batch_end = min(batch_start+batch_size, len(coco_data))

        image_embed, image_ids = embed_image_batch(model, transform_test, batch)
        embed_list.append(image_embed)
        all_ids += image_ids

        batch_start = batch_end

all_embeds = torch.cat(embed_list)
print('Embeddings shape: ' + str(all_embeds.shape))
print('Number of ids: ' + str(len(all_ids)))
torch.save({'embeddings': all_embeds, 'ids': all_ids}, 'res')

print('Finished!')
