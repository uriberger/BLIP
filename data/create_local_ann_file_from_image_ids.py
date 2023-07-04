import sys
import json
import random

assert len(sys.argv) == 2 or len(sys.argv) == 3
batch_ind = int(sys.argv[1])
image_ids_file = 'reformulation_data/batch_' + str(batch_ind) + '/batch_' + str(batch_ind) + '_image_ids.json'
with open(image_ids_file, 'r') as fp:
    image_id_list = json.load(fp)
if len(sys.argv) == 3:
    select_caption_method = sys.argv[2]
    assert select_caption_method in ['random', 'clip'], 'Unknown select caption method ' + str
else:
    select_caption_method = 'random'

if select_caption_method == 'clip':
    with open('reformulation_data/batch_' + str(batch_ind) + '/image_id_to_caption_inds.json', 'r') as fp:
        image_id_to_caption_inds = json.load(fp)

ann_data = []

with open('../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
    all_coco_data = json.load(fp)['images']

for image_id in image_id_list:
    sample = [x for x in all_coco_data if x['cocoid'] == image_id][0]
    if select_caption_method == 'random':
        caption_inds = random.sample(range(len(sample['sentences'])), 2)
    elif select_caption_method == 'clip':
        caption_inds = image_id_to_caption_inds[str(image_id)]
    for caption_ind in caption_inds:
        res = {}
        res['image_id'] = image_id
        res['image'] = sample['filepath'] + '/COCO_val2014_' + str(image_id).zfill(12) + '.jpg'
        res['caption'] = sample['sentences'][caption_ind]['raw']
        ann_data.append(res)

with open('ann.json', 'w') as fp:
    fp.write(json.dumps(ann_data))

print('Finished, loaded ' + str(len(ann_data)) + ' samples')
