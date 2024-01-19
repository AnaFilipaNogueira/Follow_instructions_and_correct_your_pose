import json

split = "test_out_sequence"
f_data = open('/nas-ctm01/datasets/public/posefix_dataset/images/data.json')
f = open('/nas-ctm01/datasets/public/posefix_dataset/posefix_release/' + split + '_pair_ids.json')

data = json.load(f_data)
file_read = json.load(f)

list_of_items = []

for i in file_read:
    #print(i)
    list_of_items.append(data[i])

f.close()
f_data.close()

json_object = json.dumps(list_of_items, indent=4)
with open("/nas-ctm01/datasets/public/posefix_dataset/" + split + ".json", "w") as outfile:
    outfile.write(json_object)

with open("/nas-ctm01/datasets/public/posefix_dataset/" + split + ".json", "w") as f1:
    data_saved = json.load(f1)

for i in data_saved:
    i['sents_0'] = str(i['uid'])
    
with open("/nas-ctm01/datasets/public/posefix_dataset/" + split + '.json') as f2:
    json.dump(data, f2, indent = 4)

f1.close()
f2.close()
