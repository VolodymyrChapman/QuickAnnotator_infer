# QuickAnnotator_infer
Infer on large numbers of images using a trained QuickAnnotator model!
Heavily borrowed from https://github.com/choosehappy/QuickAnnotator - **please cite the QuickAnnotator project if using QuickAnnotator_infer** (citation instructions on project page).

# Purpose
QuickAnnotator provides a neat, accessible and intuitive way of training Deep Learning models.
An unmet need is a simple means to infer on large numbers of images without loading them into QuickAnnotator.
This is the intended outcome of the QuickAnnotator_infer project.

A long-term goal would be an open share for clinicians/researchers across continents to try, use and compare performance of models on each others' data. Get in touch if you'd like to help take this forward!

# Requirements
---
Tested on Ubuntu 20.04 - feedback on performance on other OSs would be greatly appreciated!
The environment requirements are a cut-back version of those needed for QuickAnnotator (env files provided in the cuda10 and cuda11 directories - use whichever is suitable for your gpu).

Requires:
1. Python (tested on 3.8)
2. pip

Once pip is installed, necessary packages can be installed by navigating to the appropriate cuda folder and running:
```
pip install -r requirements.txt
```

And the following additional python package:
1. scikit_image
2. scikit_learn
3. opencv_python_headless
4. torch
5. numpy
6. matplotlib

# Usage - refer to projects/example_projects for required project folder layout
1. Create a new project folder in the 'projects' directory - feel free to add a text file describing how you created the model, types of images used etc. This will be referred to as QuickAnnotator_infer project folder
2. Copy the config file used to create your model in QuickAnnotator. Located here --> {YOUR QuickAnnotator DIRECTORY HERE}/config/config.ini
3. Paste this into your QuickAnnotator_infer project folder - we need this to recreate the model trained in QuickAnnotator
4. Copy the model you would like to infer with. These are files with the extension '.pth' located here --> {YOUR QuickAnnotator DIRECTORY HERE}/projects/{DESIRED PROJECT HERE}/models
5. Paste the '.pth' file into your QuickAnnotator_infer project folder
6. Create a folder named 'input' in your QuickAnnotator_infer project folder
7. Add images you would like to infer on with your model here (accepted file types: jpeg, jpg, png, tiff)
8. Run QuickAnnotator_infer:
```
python run_infer.py
```
9. You will be asked which project you would like to run etc. - answer in the command line
10. Inference output will appear in an 'output' directory in the QuickAnnotator_infer project folder

# Collaboration / cooperation
Forks, pull requests, co-operation all welcome!