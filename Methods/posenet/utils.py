import cv2
import numpy as np
import torch
import posenet.constants


def valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height


def _process_input(source_img, scale_factor=1.0, output_stride=16):
    target_width, target_height = valid_resolution(
        source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
    scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])

    input_img = cv2.resize(source_img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.transpose((2, 0, 1)).reshape(1, 3, target_height, target_width)
    return input_img, source_img, scale

def process_input(img, scale_factor=1.0, output_stride=16):
    return _process_input(img, scale_factor, output_stride)


def read_cap(cap, scale_factor=1.0, output_stride=16):
    res, img = cap.read()
    if not res:
        raise IOError("webcam failure")
    return _process_input(img, scale_factor, output_stride)


def read_imgfile(path, scale_factor=1.0, output_stride=16):
    img = cv2.imread(path)
    return _process_input(img, scale_factor, output_stride)


def draw_keypoints(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_confidence:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
    out_img = cv2.drawKeypoints(img, cv_keypoints, outImage=np.array([]))
    return out_img


def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1, offset = [0,0]):
    results = []
    for left, right in posenet.CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32))
    return results


def draw_skeleton(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    out_img = img
    adjacent_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_confidence)
        adjacent_keypoints.extend(new_keypoints)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img


def draw_skel_and_kp(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.5, min_part_score=0.5):

    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
        adjacent_keypoints.extend(new_keypoints)

        for ks, kc, in zip(keypoint_scores[ii, 5:11], keypoint_coords[ii, 5:11, :]):
            if ks < min_part_score:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))


    if cv_keypoints:
        out_img = cv2.drawKeypoints(
            out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img

from sklearn.linear_model import LinearRegression


# Polynomial Regression
def polyfit(x, y, degree):
    results = {}
    x -= x.mean()
    coeffs = np.polyfit(x, y, degree)
    print(x,y)
     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results

import math

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))*180/math.pi

class PoseNet():
    def __init__(self, args):
        self.model = posenet.load_model(args.model)
        print("Pose model")
        print(self.model)
        self.model = self.model.cuda()
        self.output_stride = self.model.output_stride
        self.scale_factor = args.scale_factor
        self.min_pose_score = 0.15
        self.min_part_score = 0.1
        self.max_pose_detections=10

    def process_input(self, img):
        return posenet.process_input(img, self.scale_factor, self.output_stride)



    def get_keypoints(self, img, output_scale, offset = [0,0]):
        with torch.no_grad():
            input_image = torch.Tensor(img).cuda()
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = self.model(input_image)
            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                self.output_stride,
                self.max_pose_detections,
                min_pose_score=0.015)

        keypoint_coords = keypoint_coords*output_scale + offset


        return pose_scores, keypoint_scores, keypoint_coords

    def find_signal(self, instance_scores, keypoint_scores, keypoint_coords, min_pose_score=0.5, min_part_score=0.5):
        profiles = []
        for ii, score in enumerate(instance_scores):
            if score < min_pose_score:
                continue

            signals = np.array([True, False, False, False, False, False, False])
            signal_description = ["None", "Stretched arms", "Left stretched","Right stretched","Both angles", "Angle Left", "Angle right"]
            cv_keypoints = []
            adjacent_keypoints = []

            new_keypoints = get_adjacent_keypoints(keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
            adjacent_keypoints.extend(new_keypoints)

            i = 0
            coordinate = np.zeros((6, 2))
            for ks, kc, in zip(keypoint_scores[ii, 5:11], keypoint_coords[ii, 5:11]):
                # if ks < min_part_score:
                #    continue
                cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
                coordinate[i, :] = (kc[1], kc[0])
                i += 1

            if i > 3:
                all_angles = []
                coordinate_list = [keypoint_coords[ii, 9], keypoint_coords[ii, 7], keypoint_coords[ii, 5],
                                   keypoint_coords[ii, 6], keypoint_coords[ii, 8], keypoint_coords[ii, 10]]

                total_angle = 0
                for i in range(len(coordinate_list) - 1):
                    for j in range(i + 2, len(coordinate_list)):
                        all_angles.append(abs(angle(coordinate_list[i] - coordinate_list[i + 1],
                                                    coordinate_list[j] - coordinate_list[i + 1])))

                    all_angles.append((math.atan2(coordinate_list[i][0] - coordinate_list[i + 1][0],
                                                  coordinate_list[i][1] - coordinate_list[i + 1][1]) * 180 / math.pi))
                    total_angle += abs(all_angles[-1])


                #print("Detection:",  np.std(coordinate[:, 1:2]),  np.linalg.norm(coordinate[0, :] - coordinate[1, :]) / 4, total_angle/(len(coordinate_list) - 1) )
                if np.std(coordinate[:, 1:2]) < np.linalg.norm(coordinate[0, :] - coordinate[1, :]) / 4 and total_angle/(len(coordinate_list) - 1) < 10:
                    signals[0] = False
                    signals[1] = True
                elif (abs(abs(all_angles[0]) - 180) + abs(all_angles[4]) + abs(all_angles[8])) / 3 < 10:
                    signals[0] = False
                    signals[2] = True
                elif (abs(abs(all_angles[-3]) - 180) + abs(all_angles[-2]) + abs(all_angles[-1])) / 3 < 10:
                    signals[0] = False
                    signals[3] = True
                elif abs(all_angles[0] - 90) < 20 and coordinate_list[0][0] < coordinate_list[2][0] and abs(all_angles[12] - 90) < 20 and coordinate_list[5][0] < coordinate_list[3][0]:  # abs(all_angles[9]) < 10:
                    signals[0] = False
                    signals[4] = True
                elif abs(all_angles[0] - 90) < 20 and coordinate_list[0][0] < coordinate_list[2][0]:  # abs(all_angles[9]) < 10:
                    signals[0] = False
                    signals[5] = True
                elif abs(all_angles[12] - 90) < 20 and coordinate_list[5][0] < coordinate_list[3][0]:  # abs(all_angles[13]-180) < 10:
                    signals[0] = False
                    signals[6] = True


                profiles.append((signals, signal_description, cv_keypoints, new_keypoints))

        return profiles

    def draw_profiles(self, img, profiles):
        out_img = img.copy()
        if profiles:
            for profile, i in zip(profiles,range(1,len(profiles)+1)):
                if profile[0][0]:
                    color = (0,0,255)
                else:
                    color = (0, 255, 0)
                cv2.putText(out_img,"Profile "+str(i)+" signal: ", (30, int(i*30)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                cv2.putText(out_img, profile[1][profile[0].argmax()], (350, int(i*30)), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
                cv2.putText(out_img, "Profile "+str(i), (int((profile[2][0].pt[0]+profile[2][1].pt[0])/2-50),int((profile[2][0].pt[1]+profile[2][1].pt[1])/2-30)), cv2.FONT_HERSHEY_COMPLEX, 1,color, 2)
                out_img = cv2.polylines(out_img, profile[3], isClosed=False, color=color)
                out_img = cv2.drawKeypoints(out_img, profile[2], outImage=np.array([]), color=color, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        else:
            cv2.putText(out_img, "Signal: ", (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(out_img, "None", (150, 30), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2)

        return out_img

    def check_signal(self, img, offset = [0,0]):
        input_image, display_image, output_scale = self.process_input(img)
        pose_scores, keypoint_scores, keypoint_coords = self.get_keypoints(input_image, output_scale, offset)

        return self.find_signal(pose_scores, keypoint_scores, keypoint_coords, self.min_pose_score , self.min_part_score)