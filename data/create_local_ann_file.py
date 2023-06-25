import sys
import json
import random

assert len(sys.argv) == 2
image_ids_file = sys.argv[1]
with open(image_ids_file, 'r') as fp:
    image_id_list = json.load(fp)

ann_data = []

with open('../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
    all_coco_data = json.load(fp)['images']

for image_id in image_id_list:
    sample = [x for x in all_coco_data if x['cocoid'] == image_id][0]
    caption_inds = random.sample(range(len(sample['sentences'])), 2)
    for caption_ind in caption_inds:
        res = {}
        res['image_id'] = image_id
        res['image'] = sample['filepath'] + '/COCO_val2014_' + str(image_id).zfill(12) + '.jpg'
        res['caption'] = sample['sentences'][caption_ind]['raw']
        ann_data.append(res)

with open('ann.json', 'w') as fp:
    fp.write(json.dumps(ann_data))

