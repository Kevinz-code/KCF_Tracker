# KCF_Tracker in Python
## """
Algorithms Name: Kernel Correlation Filter For Object Tracking
Author : Kevin Ke
Date : 28th, March, 2020 
"""

> [High-Speed Tracking with Kernelized Correlation Filters](http://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf)<br>

### Requirements
- Python 2.7 or 3.5
- NumPy
- OpenCV3

### Use
Download the git and
```shell
git clone https://github.com/kevin655/KCF_Tracker.git
cd KCF
python main.py
```
It will open the default camera of your computer, and the groundtruth for the first frame can be set in ./truth.txt of shape (x1, y1, x2, y2).

run
```shell
python main.py -h 
```
for parameters setting

# The following functions have been accomplished:
1. hog feature extraction
2. multi-scale prediction
3. fixed template [64, 64] for better FFT
4. gamma correct
5. roi padding
6. window margin correction
7. smooth adaptation for alpha and roi extracted
