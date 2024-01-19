import random
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import text2pose.config as config
import text2pose.demo as demo
import text2pose.utils as utils
import text2pose.utils_visu as utils_visu

import json

list_dict = []

### INPUT ################################################################################
version_human = 'posefix-H'
version_paraphrases = 'posefix-PP'
version_auto = 'posefix-A'

### SETUP ################################################################################
dataID_2_pose_info, triplet_data_human = demo.setup_posefix_data(version_human)
_, triplet_data_pp = demo.setup_posefix_data(version_paraphrases)
_, triplet_data_auto = demo.setup_posefix_data(version_auto)
pose_pairs = utils.read_json(config.file_pair_id_2_pose_ids)
body_model = demo.setup_body_model()

### DISPLAY DATA ################################################################################
# get input pair id
#pair_ID = st.number_input("Pair ID:", 0, len(pose_pairs))
print(len(pose_pairs))

#for i in range(len(pose_pairs)-1): #
pair_ID = 2580
path_img0='/home/up201705948/posescript-main/img_a{}.jpg'.format(pair_ID)
path_img1='/home/up201705948/posescript-main/img_b{}.jpg'.format(pair_ID)
dict_id = {
    "img0": path_img0,
    "img1": path_img1
}

# display information about the pair
pid_A, pid_B = pose_pairs[pair_ID]
in_sequence = dataID_2_pose_info[str(pid_A)][1] == dataID_2_pose_info[str(pid_B)][1]

# load pose data
pose_A_info = dataID_2_pose_info[str(pid_A)]
pose_A_data, rA = utils.get_pose_data_from_file(pose_A_info, output_rotation=True)
pose_B_info = dataID_2_pose_info[str(pid_B)]
pose_B_data = utils.get_pose_data_from_file(pose_B_info, applied_rotation=rA if in_sequence else None)

# render the pair under the desired viewpoint, and display it
#view_angle = st.slider("Point of view:", min_value=-180, max_value=180, step=20, value=0)
view_angle = 0
viewpoint = [] if view_angle == 0 else (view_angle, (0,1,0))
pair_img = utils_visu.image_from_pair_data(path_img0, path_img1, pose_A_data, pose_B_data, body_model, viewpoint=viewpoint, add_ground_plane=False)


# display text annotations
if pair_ID in triplet_data_human:
    for k,m in enumerate(triplet_data_human[pair_ID]["modifier"]):
        dict_id['sents_human_{}'.format(k)] = m.strip()

if pair_ID in triplet_data_pp:
    for k,m in enumerate(triplet_data_pp[pair_ID]["modifier"]):
        dict_id['sents_paraphrase_{}'.format(k)] = m.strip()

if pair_ID in triplet_data_auto:
    for k,m in enumerate(triplet_data_auto[pair_ID]["modifier"]):
        dict_id['sents_{}'.format(k)] = m.strip()

dict_id['uid'] = pair_ID
list_dict.append(dict_id)

with open("/home/up201705948/posescript-main/data1.json", "w") as file:
    json.dump(list_dict, file, indent=2)
