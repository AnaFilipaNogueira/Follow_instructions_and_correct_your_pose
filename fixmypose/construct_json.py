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