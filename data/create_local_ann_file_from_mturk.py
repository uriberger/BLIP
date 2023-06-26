import sys
import csv
import json

assert len(sys.argv) == 2
mturk_results_file = sys.argv[1]
with open(mturk_results_file, 'r') as fp:
    my_reader = csv.reader(fp)
    data = []
    for row in my_reader:
         data.append(row)
data = data[1:]

ann_data = []

for sample in data:
    if sample[29] != 'blip':
        continue
    res = {}
    image_id = int(sample[27].split('2014_')[-1].split('.jpg')[0])
    res['image_id'] = image_id
    res['image'] = 'val2014/COCO_val2014_' + str(image_id).zfill(12) + '.jpg'
    res['caption'] = sample[30]
    ann_data.append(res)

with open('ann.json', 'w') as fp:
    fp.write(json.dumps(ann_data))

print('Finished, loaded ' + str(len(ann_data)) + ' samples')
