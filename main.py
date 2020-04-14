import cv2
import numpy as np
import time
from tracker import Tracker
import argparse
import torch
import os

"""
Algorithms Name: Kernel Correlation Filter For Object Tracking
Author : Kevin Ke
Date : 28th, March, 2020 
"""

# The following functions have been accomplished:
# 1. hog feature extraction
# 2. multi-scale prediction
# 3. fixed template [64, 64] for better FFT
# 4. gamma correct
# 5. roi padding
# 6. window margin correction
# 7. smooth adaptation for alpha and roi extracted


p = argparse.ArgumentParser(description="Set Params for VOT")
p.add_argument(
    "-s",
    "--sigma",
    type=float,
    default=0.2,
    help="kernel bandwith"
)
p.add_argument(
    "-l",
    "--lamda",
    type=float,
    default=0.0001,
    help="regularation params"
)
p.add_argument(
    "-H",
    "--HOG",
    type=bool,
    default=False,
    help="whether to set hogfeature"
)
p.add_argument(
    "-M",
    "--multiscale",
    type=bool,
    default=False,
    help="multi scale prediction"
)
p.add_argument(
    "-p",
    "--pad",
    type=float,
    default=1.2,
    help="padded roi for robust tracking"
)
p.add_argument(
    "-a",
    "--adapt",
    type=float,
    default=0.075,
    help="adaptation rate for alpha and roi"
)
p.add_argument(
    "--scale",
    type=float,
    default=0.8,
    help="scale adjust rate for multiscale tracking"
)
p.add_argument(
    "--scalethresh",
    type=int,
    default=1,
    help="the horizontal and vertical pixel-shift-threshold for scale adjust"
)
p.add_argument(
    "-g",
    "--TargetGaussianBand",
    type=int,
    default=30,
    help="the bandwiths for generating gaussian targets"
)
p.add_argument(
    "--gamma",
    type=float,
    default=2.0,
    help="the gamma correct for the original image"
)


def get_groundtruth(l):
    with open("truth.txt", "r") as f1:
        first_line = f1.readlines()[l].strip()
        x1, y1, x2, y2 = map(int, first_line.split(','))
    return x1, y1, x2, y2


def get_video():
    # read the default videos
    cap = cv2.VideoCapture("2.mp4")


    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_all = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("[INFO] FPS: {:.0f}fps".format(fps))
    print("[INFO] frame_all : {:.0f}".format(frame_all))
    print("[INFO] time_last: {:.1f}s".format(frame_all / fps))

    return cap

id = 1
os.remove("results.txt")
def main(argv):
    sigma = argv["sigma"]
    lamda = argv["lamda"]
    hog = argv["HOG"]
    multiscale = argv["multiscale"]
    pad = argv["pad"]
    adapt = argv["adapt"]
    scale = argv["scale"]
    scale_thresh = argv["scalethresh"]
    TargetGaussianBand = argv["TargetGaussianBand"]
    gamma = argv["gamma"]

    # init
    cap = get_video()
    init_tracker = True
    x1, y1, x2, y2 = get_groundtruth(0)
    kcf = Tracker(sigma=sigma, lamda=lamda, hog=hog, multiscale=multiscale,
                  pad=pad, adapt=adapt, scale=scale, scale_thresh=scale_thresh,
                  TargetGaussianBand=TargetGaussianBand, gamma=gamma)
    time_last = 0.0

    while 1:
        ret, frame = cap.read()
        # Set to numpy array and to raw pixels
        frame_cal = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if (init_tracker):
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 255))

            # init Tracker
            kcf.set_first_frame(frame_cal, [x1, y1, x2, y2])
            init_tracker = False

        else:
            global id

            # Refresh frame
            t1 = time.time()
            new_region = kcf.refresh(cur_frame=frame_cal)
            t2 = time.time()
            duration = (1 - 0.1) * time_last + 0.1 * float(t2 - t1)  # exp mean
            time_last = duration
            print("%.3f fps" % (1.0 / duration))

            # set rectangle
            x1, y1, x2, y2 = new_region
            """
            if x1 < 0:
                x2 -= x1
                x1 = 0
            if y1 < 0:
                y2 -= y1
                y1 = 0
            """
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), thickness=3)
            #frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cv2.putText(frame, 'FPS: ' + str(1.0 / duration)[:5], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
            cv2.imshow("Kevin's Tracker", frame)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        line = "%d,%d\n"%(cx, cy)
        with open("results.txt", "a") as f:
            f.writelines(line)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


argv = vars(p.parse_args())
main(argv=argv)

