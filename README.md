# Eye contact detection using Generative Synthesized Image based Gaze Estimation
This repository contains the implementation of the research project.

## Requirements
``` python==3.5.6 ```
All requirements can be installed with ``` python -m pip install -r requirements.txt ```

## Image generation and gaze redirection
Uncomment the "Train gaze redirection model" part of code from ```main.py``` file to train the gaze redirection network.

Uncomment the "Generate redirected images" part of code from ```main.py``` file to generate images from the trained gaze redirection network.

## Create h5 files
Uncomment the "Convert dataset directory to h5 file" part of code from ```main.py``` file to generate h5 files from dataset directory.

## Gaze estimation
Run the ```main.py``` with the required train and test h5 files in "Train and evaluate DPG Gaze estimation" part to evaluate the gaze estimator.
