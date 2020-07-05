from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from YOLO.util import *
import argparse
import os
import os.path as osp
from YOLO.darknet import Darknet
import pickle as pkl
import pandas as pd
import random
from SiameseNetwork import *
import torch.nn.functional as F

def arg_parse():
    """
    Parse arguements to the detect module
    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.9)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="YOLO/cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="YOLO/yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument("--video", dest="videofile", help="Video file to     run detection on", default="video.avi",
                        type=str)

    return parser.parse_args()

def init_YOLO(args):
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    CUDA = torch.cuda.is_available()
    print("Cuda:", CUDA)

    num_classes = 80
    classes = load_classes("YOLO/data/coco.names")

    # Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()

    return model, CUDA, batch_size, confidence, nms_thesh, num_classes, classes , inp_dim

def write(x, results, pre_bb, pre_color, frame, pre_frame, id_model):
    colors = pkl.load(open("pallete", "rb"))
    classes = load_classes('data/coco.names')
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])

    bb_iou_max = 0
    index_max = 0


    frame = frame[int(x[2]):int(x[4]), int(x[1]):int(x[3])]


    #frame = cv2.resize(frame,(100,200))
    cv2.imshow("t3", frame)
    frame = prepae_frame(frame)
    frame = frame.cuda()

    euclidean_distance_min = 100
    output = id_model.forward_once(frame.unsqueeze(0))
    if not not pre_frame:
        for bb, pf, index in zip(pre_bb, pre_frame, range(len(pre_bb))):
            temp = bbox_iou(x[None, 1:5], bb[None, 1:5])
            euclidean_distance = F.pairwise_distance(output, pf)
            if euclidean_distance.item() < euclidean_distance_min:
                bb_iou_max = temp
                euclidean_distance_min = euclidean_distance.item()
                index_max = index
        print("Distance:", euclidean_distance_min)

    if euclidean_distance_min < 10:
        color = pre_color[index_max]
    else:
        color = random.choice(colors)

    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)

    t_size = cv2.getTextSize(label + str(color[0]), cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label + " " + str(color[0]), (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                [225, 255, 255], 1);
    return [img, color, output, euclidean_distance_min]


# Detection phase

if __name__ == "__main__":
    args = arg_parse()
    model, CUDA, batch_size, confidence, nms_thesh, num_classes, classes, inp_dim = init_YOLO(args)

    videofile = args.videofile  # or path to the video file.
    cap = cv2.VideoCapture(videofile)
    cap = cv2.VideoCapture(0)  # for webcam
    assert cap.isOpened(), 'Cannot capture source'

    frames = 0
    start = time.time()
    pre_bb = []
    pre_color = []
    pre_frame = None

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            img = prep_image(frame, inp_dim)
            #        cv2.imshow("a", frame)
            #        cv2.waitKey(100)
            im_dim = frame.shape[1], frame.shape[0]
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = model(Variable(img, volatile=True), CUDA)
            output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.4f}".format(frames / (time.time() - start)))
                cv2.imshow("frame", frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
            output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

            output[:, 1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))
            print(colors)
            temps = list(map(lambda x: write(x, frame, pre_bb, pre_color), output))

            pre_bb = output
            pre_color = []
            for temp in temps:
                pre_color.append(temp[1])

            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print(time.time() - start)
            print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
        else:
            break






