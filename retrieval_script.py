import torch
import time
import json
import torch.nn.functional as F

image_embeddings = torch.load('COCO_test_image_embeddings')
text_embeddings = torch.load('COCO_test_text_embeddings')
image_id_to_ind = {}
for i in range(len(image_embeddings['image_ids'])):
    image_id_to_ind[image_embeddings['image_ids'][i]] = i
text_id_to_ind = {}
for i in range(len(text_embeddings['ids'])):
    text_id_to_ind[text_embeddings['ids'][i]] = i

def compute_sim(texts, image_ids):
    with torch.no_grad():
        text_inds = [text_id_to_ind[x] for x in texts]
        image_inds = [image_id_to_ind[x] for x in image_ids if x in image_id_to_ind]
        text_embed = text_embeddings['embeddings'][text_inds]
        image_embed = image_embeddings['embeddings'][image_inds]
        res = torch.matmul(text_embed, image_embed.transpose(1, 0))
    return res

with open('../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
    coco_data = json.load(fp)

coco_data = [x for x in coco_data['images'] if x['split'] == 'test']

correct_count = 0
count = 0
res = {}
all_image_ids = [x['cocoid'] for x in coco_data if x['cocoid'] in image_id_to_ind]
t = time.time()
for i in range(len(coco_data)):
    if i % 100 == 0:
        print('Starting sample ' + str(i) + ' out of ' + str(len(coco_data)) + ', time from prev ' + str(time.time()-t), flush=True)
        t = time.time()
    image_id = coco_data[i]['cocoid']
    if image_id not in image_id_to_ind:
        continue
    correct_ind = image_id_to_ind[image_id]
    res[image_id] = []
    for j in range(len(coco_data[i]['sentences'])):
        orig_caption_id = coco_data[i]['sentences'][j]['sentid']
        sim_mat = compute_sim([orig_caption_id], all_image_ids)
        selected_ind = torch.argmax(sim_mat).item()
        if selected_ind == correct_ind:
            correct_count += 1
        count += 1
        res[image_id].append(sim_mat)

accuracy = correct_count/count
print(accuracy)
torch.save(res, 'res')
print('Finished!')
