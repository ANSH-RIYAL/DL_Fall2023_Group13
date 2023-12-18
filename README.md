# DL_Fall2023_Group13
 Code Submission for Final project - Deep Learning (NYU MSDS)
<br><br>
Results:<br> 
Final IOU on Validation Dataset: 0.2166<br>
Final IOU on Hidden Dataset: 0.2209
<br><br>
Steps for replication:<br>
Clone repo, download all datasets into the repository.<br>
Open Terminal and navigate to repo, then:<br>
1. mkdir checkpoints<br>
<br>
Ensure to set base_dir variable in datasets.py to the dataset directory.<br>
<br>
To train frame prediction model:<br>
2. python3 frame_prediction.py <br>
This will save weights to the checkpoints folder frame_prediction.pth<br>
<br>
To train segmentation model:<br>
3. python3 segmentation_mask.py<br>
This will save weights to the checkpoints folder as image_segmentation.pth<br>
4. python3 hidden_eval.py<br>
This will load weights from checkpoints folder and run inference on the hidden/test set.<br>
<br>
This will create a file 'leaderboard_2_team_13.pt'<br>
This file contains a tensor of shape (2000,160,240) representing the mask for each of the 2000 videos in the hidden dataset.

Note: We referred to the following github repositories while working on this project
https://github.com/milesial/Pytorch-UNet
https://github.com/simmimak/Video-Frame-Prediction-and-Semantic-Segmentation-with-SSL/tree/main

