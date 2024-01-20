# Follow instructions and correct your pose
## Directories
The PoseFix directory was imported from the original repositories and some files were adapted or created to be able to generate the human meshes images from the available keypoints and the .json file with the necessary information, then, this .json file was divided in various files, one for each split, using the file in fixmypose/construct_json.py.

<p float="left">
  <img src="https://github.com/AnaFilipaNogueira/Follow_instructions_and_correct_your_pose/blob/main/img_a0_new.jpg" width=25% height=50% />
  <img src="https://github.com/AnaFilipaNogueira/Follow_instructions_and_correct_your_pose/blob/main/img_b0_new.jpg" width=25% height=50% />
  <img src="https://github.com/AnaFilipaNogueira/Follow_instructions_and_correct_your_pose/blob/main/description.png" width=40% height=100% />
</p></br>
<em>Fig.1 - Example of a human mesh pair</em><br/><br/>

The FixMyPose directory was imported from the original project, however, some files were changed or created to suit the modifications made, namely, regarding the possibility of using or not the pre-training from ImageNet, the addition of the data augmentation techniques: CutOut, Spelling Augmenter, Random Deletion, Synonym Replacement, Sometimes and Sequential, and also, the adaption to allow to run the model using the PoseFix dataset.
