import argparse
import time
import torch
import json
from predict import Predictor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='COCO', choices=['COCO', 'flickr30k', 'xm3600'])
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--model_path', required=True)
    args = parser.parse_args()

    with open(args.input_file, 'r') as fp:
        data = json.load(fp)
    if type(data[0]) is dict:
        image_ids = [x['image_id'] for x in data]
    else:
        image_ids = data

    if args.dataset == 'COCO':
        with open('../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
            coco_data = json.load(fp)['images']
        iid_to_split = {x['cocoid']: 'train' if x['split'] == 'train' else 'val' for x in coco_data}
            
    output_file_name = args.output_file

    model_path = args.model_path
    predictor = Predictor()
    print("Setting up predictor", flush=True)
    predictor.setup(model_path, task='image_captioning')

    print("Generating captions", flush=True)
    res = []
    t = time.time()
    for i in range(len(image_ids)):
        if i % 100 == 0:
            print(f'Starting sample {i} out of {len(image_ids)}, time from prev {time.time() - t}', flush=True)
            t = time.time()
            with open(args.output_file, 'w') as fp:
                fp.write(json.dumps(res))

        image_id = image_ids[i]
        if args.dataset == 'COCO':
            image_path = f'/cs/labs/oabend/uriber/datasets/COCO/{iid_to_split[image_id]}2014/COCO_{iid_to_split[image_id]}2014_{str(image_id).zfill(12)}.jpg'
        elif args.dataset == 'flickr30k':
            image_path = f'/cs/labs/oabend/uriber/datasets/flickr30/images/{image_id}.jpg'
        elif args.dataset == 'xm3600':
            image_path = f'/cs/labs/oabend/uriber/datasets/crossmodal3600/images/{hex(image_id)[2:].zfill(16)}.jpg'
        else:
            assert False, f'Unknown dataset {args.dataset}'

        with torch.no_grad():
            caption = predictor.predict(image=image_path, question=None, caption=None)
        res.append({'image_id': image_id, 'caption': caption})

    with open(args.output_file, 'w') as fp:
        fp.write(json.dumps(res))

