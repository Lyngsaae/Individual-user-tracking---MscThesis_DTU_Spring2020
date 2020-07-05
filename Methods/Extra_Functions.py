import cv2
import numpy as np

# Draw bonding box and pose
def drawTracking(frame, c1, c2, PoseNet):
    x_offset = int(abs(c1[0] - c2[0]) * 0.5)
    if c1[0] - x_offset < 0:
        x_offset = c1[0]

    if c2[0] + x_offset >= frame.shape[1]:
        x_offset = frame.shape[1] - c2[0] - 1

    profiles = PoseNet.check_signal(frame[int(c1[1]):int(c2[1]), int(c1[0] - x_offset):int(c2[0] + x_offset)], (c1[1], c1[0] - x_offset))
    if profiles:
        profile_signal = profiles[0]
    else:
        profile_signal = [[True, False, False, False, False, False]]
    frame = cv2.rectangle(frame, c1, c2, (255,0,0), 1)

    if profiles and False:
        if profile_signal[0][0]:
            color_pos = (0, 0, 255)
        else:
            color_pos = (0, 255, 0)
        frame = cv2.polylines(frame, profile_signal[3], isClosed=False, color=color_pos)
        frame = cv2.drawKeypoints(frame, profile_signal[2], outImage=np.array([]), color=color_pos, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    t_size = cv2.getTextSize("Tracking", cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c1 = tuple((int((c1[0] + c2[0] - t_size[0]) / 2), c1[1] - (t_size[1] + 4)))
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    frame = cv2.rectangle(frame, c1, c2, (255,0,0), -1)
    frame = cv2.putText(frame, "Tracking", (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255],1);

    return frame, profile_signal[0]


# Calculate IoU
def bb_intersection_over_union(boxA, boxB, change_format=True):
    if change_format:
        boxA[2] += boxA[0]
        boxA[3] += boxA[1]
        boxB[2] += boxB[0]
        boxB[3] += boxB[1]

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# Find new bounding box, with optimal overlap
def findNewBox(boxA, boxB, change_format=True):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    width = int(abs(max((xB - xA, 0)))/2)
    height = int(abs(max((yB - yA), 0))/2)

    newBox1 = boxB.copy()
    if boxA[0] <= boxB[0] and boxA[2] >= boxB[0]:
        if boxA[2] <= boxB[2]:
            newBox1[0] += width
    elif boxA[0] <= boxB[2] and boxA[2] >= boxB[2]:
        if boxA[0] >= boxB[0]:
            newBox1[2] -= width

    newBox2 = boxB.copy()
    if boxA[1] <= boxB[1] and boxA[3] >= boxB[1]:
        if boxA[3] <= boxB[3]:
            newBox2[1] += height
    elif boxA[1] <= boxB[3] and boxA[3] >= boxB[3]:
        if boxA[1] >= boxB[1]:
            newBox2[3] -= height

    if newBox1 != boxB and newBox2 != boxB:
        return newBox1 if (newBox1[2] - newBox1[0]) * (newBox1[3] - newBox1[1]) > (newBox2[2] - newBox2[0]) * (newBox2[3] - newBox2[1]) else newBox2
    elif newBox1 != boxB:
        return newBox1
    elif newBox2 != boxB:
        return newBox2
    else:
        return boxB

