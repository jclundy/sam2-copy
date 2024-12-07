# CSC2529
## Relevant files
See the following files:

- notebooks/foosball_angle_estimation.ipynb
- notebooks/foosball_training_set_labeller.ipynb
- video_labeller_utils.py

## Data labelling

`notebooks/foosball_training_set_labeller.ipynb` uses sam2 to generate masks on objects across the training video.  These masks are then used to generate bounding boxes and label files for a training data set.

## Angle estimation
`notebooks/foosball_angle_estimation.ipynb` uses sam2 to generate bounding boxes on the input video, then runs the angle estimation algorithm on the bounding boxes.  It loads the pre-computed 'ground-truth' angle data computed in [ground_truth_rotations_from_aruco.ipynb](https://github.com/jclundy/csc2529-project/blob/master/ground_truth_rotations_from_aruco.ipynb) for the regression and error analysis.
