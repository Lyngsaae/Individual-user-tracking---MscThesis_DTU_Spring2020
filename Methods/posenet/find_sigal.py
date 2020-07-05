import torch
import cv2
import time
import argparse
import sklearn.linear_model as linear_model
import numpy as np

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
args = parser.parse_args()


#Option
record = True

def main():
    model = posenet.PoseNet(args)

    cap = cv2.VideoCapture(r"C:\Users\Ma-Ly\OneDrive\DTU\Elektroteknologi Kandidat\4. semester\Speciale\Code\Project\PersonTrackingData\2_person_difficult.mp4")
    #cap = cv2.VideoCapture(args.cam_id)
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)

    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height =int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20#int(cap.get(cv2.CAP_PROP_FPS))

    start = time.time()
    frame_count = 0

    print("Image Widt", img_width, "\tImage Height", img_height, "\tFrames per second", fps)
    if record:
        timestamp = time.strftime('%b_%d_%Y_%H%M', time.localtime())
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter('Results/output' + timestamp + '.mp4', fourcc, fps, (img_width, img_height))

    while True:
        res, img = cap.read()
        if not res:
            raise IOError("webcam failure")

        profiles = model.check_signal(img)
        overlay_image = model.draw_profiles(img,profiles)
        cv2.imshow('posenet', overlay_image)
        #cv2.waitKey()
        frame_count += 1


        if record:
            video.write(overlay_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if record:
        video.release()
        print("Saved: Output_"+timestamp)

    print('Average FPS: ', frame_count / (time.time() - start))

if __name__ == "__main__":
    main()