# QuickAnnotator_infer
Infer on large numbers of images using a trained QuickAnnotator model!
Heavily borrowed from https://github.com/choosehappy/QuickAnnotator - please cite the QuickAnnotator project if using this repo (citation instructions on project page).

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