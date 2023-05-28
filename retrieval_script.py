import torch
import os
import random
import time
import json
import torch.nn.functional as F
from models.blip_retrieval import blip_retrieval
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

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

def compute_sim(model, transform_test, texts, image_paths):
    device = torch.device('cuda')
    with torch.no_grad():
        text = texts
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))

        image_list = []
        for image_path in image_paths:
            image = Image.open(image_path)
            image = transform_test(image)
            image = image.unsqueeze(dim=0)
            image = image.to(device)
            image_list.append(image)
        
        images = torch.cat(image_list)
        image_feat = model.visual_encoder(images)
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)

        res = torch.matmul(text_embed, image_embed.transpose(1, 0))
    return res

def get_all_others(i, sample_num):
    res = list(range(sample_num))
    res.pop(i)
    return res

with open('../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
    coco_data = json.load(fp)

coco_data = [x for x in coco_data['images'] if x['split'] == 'val']

print('Loading model...', flush=True)
model, transform_test = load_model()

correct_count = 0
count = 0
t = time.time()
#for i in range(len(coco_data)):
for i in range(100):
    if i % 2 == 0:
        print('Starting sample ' + str(i) + ' out of ' + str(len(coco_data)) + ', time from prev ' + str(time.time()-t), flush=True)
        t = time.time()
    orig_caption = random.choice([x['raw'] for x in coco_data[i]['sentences']])
    distractor_indices = get_all_others(i, len(coco_data))
    option_indices = [i] + distractor_indices
    image_paths = [os.path.join('/cs/labs/oabend/uriber/datasets/COCO', coco_data[index]['filepath'], coco_data[index]['filename']) for index in option_indices]

    texts = [orig_caption]
    sim_mat = compute_sim(model, transform_test, texts, image_paths)
    selected = torch.argmax(sim_mat).item()
    if selected == 0:
        correct_count += 1
    count += 1

accuracy = correct_count/count
print(accuracy)
