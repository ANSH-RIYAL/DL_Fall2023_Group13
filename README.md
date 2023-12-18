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
1. mkdir checkpoints
3. python3 frame_prediction.py
4. python3 segmentation_mask.py
5. python3 hidden_eval.py

This will create a file 'leaderboard_2_team_13.pt'<br>
This file contains a tensor of shape (2000,160,240) representing the mask for each of the 2000 videos in the hidden dataset.
