# DL_Fall2023_Group13
 Code Submission for Final project - Deep Learning (NYU MSDS)

Results: 
Final IOU on Validation Dataset: 0.2166
Final IOU on Hidden Dataset: 0.2209

Steps for replication:
Clone repo, download all datasets into the repository.
Open Terminal and navigate to repo, then:
1. mkdir checkpoints
3. python3 frame_prediction.py
4. python3 segmentation_mask.py
5. python3 hidden_eval.py

This will create a file 'leaderboard_2_team_13.pt'
This file contains a tensor of shape (2000,160,240) representing the mask for each of the 2000 videos in the hidden dataset.
