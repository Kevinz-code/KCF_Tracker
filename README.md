# KCF_Tracker in Python
 
Algorithms Name: Kernel Correlation Filter For Object Tracking
Author : Kevin Ke
Date : 28th, March, 2020 

> [High-Speed Tracking with Kernelized Correlation Filters](http://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf)<br>

## The following functions have been accomplished:
- hog feature extraction
- multi-scale prediction
- fixed template [64, 64] for better FFT
- gamma correct
- roi padding
- window margin correction
- smooth adaptation for alpha and roi extracted

### Requirements
- Python 2.7 or 3.5
- NumPy
- OpenCV3

### Use
Download the git and
```shell
git clone https://github.com/Kevinz-code/KCF_Tracker.git
cd KCF
python main.py
```
It will open the default camera of your computer, and the groundtruth for the first frame can be set in ./truth.txt of shape (x1, y1, x2, y2).

for parameters setting, run
```shell
python main.py -h 
```

## Problems
  I'm a newer in DeepLearning, so the code may not reach the same results as the orginal paper. For example, HOG feature based tracking in this repository does not work well and I'm fixing it right now. Multi_scale tracking is also not stable.
  Fortunately, the raw pixels based tracking is rather stable and can be very useful for beginners in tracking area.

