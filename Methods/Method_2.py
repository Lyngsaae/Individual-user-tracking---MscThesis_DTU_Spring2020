from __future__ import division

import time
import glob
from ADNet.ADNet import *
from YOLO.YOLO import *
torch.cuda.current_device()
import posenet
from SiameseNetwork.SiameseNetwork import *
from Extra_Functions import *

# YOLO setup
args = arg_parse()
model, CUDA, batch_size, confidence, nms_thesh, num_classes, classes, inp_dim = init_YOLO(args)
model.eval()


# ReID setup
id_path = r"C:/Users/Ma-Ly/Google Drev/DTU - Speciale F2020/Siamese_Networks/models/QaudrupletHardNet_Feb_26_2020_1242.pt"
id_model = load_model(id_path, 1)
id_model.eval()

#Action decision network set up
action_amount = 17
pre_action_amount = 10
AD_path = r"C:/Users/Ma-Ly/Google Drev/DTU - Speciale F2020/Action_Decision_Network/Models/ADNet_SL_Advanced_Mar_30_2020_1156.pt"#ADNet_SL_Mar_19_2020_1215.pt"
AD_model = ADNet(action_amount,pre_action_amount).cuda()
AD_model.load_state_dict(torch.load(AD_path))
AD_model.eval()


transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((100,50)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                ])

agent = ADagent(AD_model, transform, 2)


# PoseNet setup
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=1)
args = parser.parse_args()

PoseNet = posenet.PoseNet(args)
PoseNet.max_pose_detections = 1
PoseNet.min_part_score = 0
PoseNet.min_pose_score = 0.3


# Options
record = False
start_funcion = True
print_info = False
save_bb = False
save_stats = False

samples = [0,
            "Test_data/test_video.mp4",
           "Test_data/1_person_easy.mp4",
           "Test_data/1_person_difficult.mp4",
           "Test_data/2_person_easy.mp4",
           "Test_data/2_person_difficult.mp4",
           "Test_data/20200518_100805.mp4",
           "Test_data/20200519_214042.mp4",
           "Test_data/20200519_214131.mp4",
           "Test_data/20200519_214325.mp4",
           "Test_data/20200519_214554.mp4",
           "Test_data/20200519_214720.mp4",
           "Test_data/20200519_214820.mp4",
           "Test_Data/20200519_214847.mp4",
           "Test_data/20200519_214910.mp4",
           "Test_data/20200603_205814.mp4",
           "Test_data/20200603_205901.mp4",
           "Test_data/20200603_205942.mp4",
           "Test_data/20200603_210036.mp4",
           "Test_data/20200603_210109.mp4",
           "Test_data/20200603_210228.mp4",
           "Test_data/20200603_210319.mp4"]

rotations = [0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,2,2,2,2,2,2,2]

#video_name = samples[9]

for video_index in range(2,22):
    if video_index == 2 or video_index == 4 or video_index == 10 or video_index == 11: #Videos where start signal is not detected
        continue
    video_name = samples[video_index]

    # Initiliaze variables
    cap = cv2.VideoCapture(video_name)
    assert cap.isOpened(), 'Cannot capture source'

    path = r"C:/Users/Ma-Ly/Google Drev/DTU - Speciale F2020/Action_Decision_Network/Data/Training/Human2/"
    files = glob.glob(path + '*.jpg')#os.listdir(path)
    files.sort()
    file_counter = 0



    found_flag = False
    wait_flag = False
    start_flag = False
    lock_flag = False
    pre_feature_vectors = []
    pre_fake_feature_vectors = []
    AD_bb = None
    signal_counter = 0
    found_count = 0
    YOLO_count = 0
    YOLO_tracking_counter = 0
    YOLO_tracking_found_counter = 0
    ADNet_tracking_counter = 0
    redetect_counter = 0
    total_track_counter = 0
    redect = False
    frame_counter = 0
    pre_length = 0
    AD_count = 0
    threshold = 5
    image_threshold = 3
    frame_scale = 1
    yolo_frame_scale = 0.2
    ad_confidence = False
    signal_start = False
    signal_counter = 0
    YOLO = True
    pre_objects = 0
    count_n_detect = 0
    stop_counter = 0
    IoU_condition = True

    # Set up recorder
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*frame_scale)
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*frame_scale)

    fps = 24#int(cap.get(cv2.CAP_PROP_FPS))
    print("Image Widt", img_width, "\tImage Height", img_height, "\tFrames per second", fps)
    if not isinstance(video_name, int):
        timestamp = time.strftime('%b_%d_%Y_%H%M', time.localtime())
        name = os.path.splitext(os.path.basename(video_name))[0]

        if record:
            fourcc = cv2.VideoWriter_fourcc(*'FMP4')
            video = cv2.VideoWriter('Results/Method 2/' + name + "_"+ timestamp + '.mp4', fourcc, fps, (img_width, img_height))

    if save_bb:
        name = os.path.splitext(os.path.basename(video_name))[0]
        bb_file= open("Results/Method 2/bounding boxes/"+name+"_"+timestamp+".txt","w+")

    if save_stats and not 'stats_file' in locals():
        name = os.path.splitext(os.path.basename(video_name))[0]
        stats_file= open("Results/Method 2/Statistics/stats_"+timestamp+".txt","w+")
        stats_file.write("Video name , YOLO Trackings , YOLO trackings found, ADNet Trackings , Re-detections, Track frames , Frames per second , ADNet time , YOLO time, Frames\n")


    start_time = time.time()
    ADNet_time = 0
    YOLO_time = 0

    # Start executing input
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            if rotations[video_index] == 1:
                frame = cv2.flip(frame, -1)
            elif rotations[video_index] == 2:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame_counter += 1
            if lock_flag:
                total_track_counter += 1
            frame = cv2.resize(frame, (int(frame.shape[1] * frame_scale), int(frame.shape[0] * frame_scale)))
            ad_frame = frame.copy()
            im_dim = int(frame.shape[1]*yolo_frame_scale), int(frame.shape[0]*yolo_frame_scale)
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)
            flag_2 = False


            if start_funcion and not lock_flag:
                ad_confidence = False


            yolo_flag = (not ad_confidence or YOLO_count < 2 and len(pre_feature_vectors) > 5) and AD_count == 0 or signal_start or start_funcion and not lock_flag
            if not yolo_flag:
                if lock_flag:
                    ADNet_tracking_counter += 1
                    start_time_temp = time.time()
                YOLO_count  = 0
                redect = False
                scalar = 1
                YOLO = False
                AD_bb, ad_confidence = agent.takeAction(ad_frame, AD_bb)
                id_frame = ad_frame[ AD_bb[1]: AD_bb[1] + AD_bb[3],  AD_bb[0]: AD_bb[0] +  AD_bb[2], :]
                id_frame = prepae_frame(id_frame).cuda()

                with torch.no_grad():
                    feature_vector = id_model(id_frame.unsqueeze(0))

                condition, dst, min_dst, fake_dst, mean_dst = test_similarity(feature_vector, pre_feature_vectors, pre_fake_feature_vectors, max((threshold,3.5)))

                #print("Runing ADNet - Confidence:", ad_confidence, " Condition:", condition, threshold, dst, min_dst, fake_dst, mean_dst)
                #cv2.waitKey()
                if dst < threshold  and (lock_flag or not start_funcion) and condition:
                    pre_feature_vectors.append(id_model(id_frame.unsqueeze(0)))
                    if len(pre_feature_vectors) > 20:
                        pre_feature_vectors.pop(2)

                if not condition:
                    ad_confidence = False

                if ad_confidence and AD_count < 3:
                    AD_count += 1
                elif not ad_confidence and AD_count > 0:
                    AD_count -= 1

                if lock_flag:
                    ADNet_time += time.time()-start_time_temp



            if yolo_flag or not ad_confidence:
                if lock_flag:
                    YOLO_tracking_counter += 1
                    start_time_temp = time.time()
                    if not redect:
                        redetect_counter += 1
                redect = True
                scalar = 0.5
                YOLO = True
                img = cv2.resize(frame,(int(frame.shape[1] * yolo_frame_scale), int(frame.shape[0] * yolo_frame_scale)))
                img = prep_image(img, inp_dim)
                if CUDA:
                    im_dim = im_dim.cuda()
                    img = img.cuda()

                with torch.no_grad():
                    output = model(Variable(img, volatile=True), CUDA)

                #print("Running YOLO")
                output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

                # Init variables
                min_dst_total = 1000
                img = frame
                flag = False
                second_dst = 1000
                condition = False
                index = -1
                temp_fature_vectors = []
                mean_dst_buffer = []
                fake_dst_bufer = []
                i = -1
                pre_signal_start = bool(signal_start)
                signal_start = False
                AD_bb = None
                c1 = [-1, -1]
                c2 = [-1, -1]

                # If not object is found
                if type(output) != int:
                    #Scale boxes to fit original image
                    im_dim = im_dim.repeat(output.size(0), 1)
                    scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)
                    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
                    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2
                    output[:, 1:5] /= scaling_factor

                    # Fit box sizes to image
                    for j in range(output.shape[0]):
                        output[j, [1, 3]] = torch.clamp(output[j, [1, 3]], 0.0, im_dim[j, 0])
                        output[j, [2, 4]] = torch.clamp(output[j, [2, 4]], 0.0, im_dim[j, 1])


                    if pre_objects != len(output):
                        pre_objects = len(output)
                        update_counter = 0
                    else:
                        update_counter += 1

                    #print("Potnetial users:", len(output))
                    for x in output:

                        if start_funcion and (not lock_flag or not start_funcion) and (abs(x[2] - x[4]) < img_height * 0.1 * yolo_frame_scale or abs(x[1] - x[3]) < img_width * 0.05 * yolo_frame_scale):  # Set according to frame size
                            continue


                        i += 1
                        # Set box size and name
                        c1 = tuple((x[1:3].cpu().numpy() / yolo_frame_scale).astype(np.int))
                        c2 = tuple((x[3:5].cpu().numpy() / yolo_frame_scale).astype(np.int))

                        ratio =  abs(c2[1].item() - c1[1].item()) / abs(c2[0].item() - c1[0].item())
                        AD_bb_temp = [c1[0],c1[1], c2[0]-c1[0], c2[1]-c1[1]]
                        id_frame = ad_frame[c1[1]:c2[1], c1[0]:c2[0]]
                        id_frame = prepae_frame(id_frame).cuda()

                        with torch.no_grad():
                            feature_vector = id_model(id_frame.unsqueeze(0))

                        condition, dst, min_dst, fake_dst, mean_dst = test_similarity(feature_vector, pre_feature_vectors, pre_fake_feature_vectors, threshold)


                        temp_fature_vectors.append(feature_vector)
                        mean_dst_buffer.append(mean_dst)
                        fake_dst_bufer.append(fake_dst)

                        if start_funcion and  (not lock_flag or pre_signal_start or len(pre_feature_vectors) <= 5):
                            x_offset = int(abs(float(c1[0] - c2[0])) * 0.5)

                            if c1[0] - x_offset < 0:
                                x_offset = c1[0]

                            if c2[0] + x_offset >= frame.shape[1]:
                                x_offset = frame.shape[1] - c2[0] - 1

                            profiles = PoseNet.check_signal(frame[int(c1[1]):int(c2[1]), int(c1[0] - x_offset):int(c2[0] + x_offset)],  (c1[1], c1[0] - x_offset))


                            for profile in profiles:
                                if profile[0][1]  or count_n_detect > 20 and (profile[0][5] or profile[0][6]) and lock_flag:
                                    AD_bb = AD_bb_temp
                                    index = i
                                    signal_start = True


                        if not start_funcion and not signal_start and len(pre_feature_vectors) <= 5:
                            AD_bb = AD_bb_temp
                            index = i
                            signal_start = True


                        if len(pre_feature_vectors) > 5 and (not signal_start or lock_flag):
                            ad_confidence = True

                            x_offset = int(abs(float(c1[0] - c2[0])) * 0.5)

                            if c1[0] - x_offset < 0:
                                x_offset = c1[0]

                            if c2[0] + x_offset >= frame.shape[1]:
                                x_offset = frame.shape[1] - c2[0] - 1

                            profiles = PoseNet.check_signal(frame[int(c1[1]):int(c2[1]), int(c1[0] - x_offset):int(c2[0] + x_offset)], (c1[1], c1[0] - x_offset))
                            if dst < min_dst_total and ad_confidence and condition or min_dst < fake_dst and count_n_detect > 10 and update_counter > 5:
                                min_dst_total = dst
                                AD_bb = [c1[0].copy(), c1[1].copy(), abs(c1[0] - c2[0]), abs(c1[1] - c2[1])]
                                index = i
                            elif profiles:
                                if  lock_flag and min_dst < 4 and (profiles[0][0][5] or profiles[0][0][6] ):
                                    min_dst_total = -1
                                    AD_bb = [c1[0].copy(), c1[1].copy(), abs(c1[0] - c2[0]), abs(c1[1] - c2[1])]
                                    index = i
                                    pre_fake_feature_vectors = []

                    if lock_flag or not start_funcion:
                        if index != -1:
                            IoU_condition = True
                            c1 = AD_bb[:2]
                            c2 = [AD_bb[0]+AD_bb[2], AD_bb[1]+AD_bb[3]]
                            for x, i in zip(output, range(len(output))):
                                # If bb is too small ignore
                                if start_funcion and (not lock_flag or not start_funcion) and (
                                        abs(x[2] - x[4]) < img_height * 0.2 or abs(x[1] - x[3]) < img_width * 0.1) or i == index:  # Set according to frame size
                                    continue
                                # Set box size and name
                                IoU = bb_intersection_over_union((x[1:5].cpu().numpy()/yolo_frame_scale).astype(np.int),[c1[0],c1[1],c2[0],c2[1]],False)
                                if IoU > 0.05:
                                    IoU_condition = False
                                    bb = (x[1:5].cpu().numpy() / yolo_frame_scale).astype(np.int).tolist()
                                    newBB = findNewBox([c1[0], c1[1], c2[0], c2[1]], bb, False)

                                    if newBB != bb:
                                        id_frame = frame[newBB[1]: newBB[3], newBB[0]: newBB[2]]
                                        id_frame = prepae_frame(id_frame).cuda()
                                        with torch.no_grad():
                                            feature_vector = id_model(id_frame.unsqueeze(0))
                                        pre_fake_feature_vectors.extend(feature_vector)
                                else:
                                    pre_fake_feature_vectors.extend(temp_fature_vectors[i])
                        else:
                            for x1, i in zip(output, range(len(output))):
                                IoU_fake_condition = True
                                for x2, j in zip(output, range(len(output))):
                                    if i != j:
                                        IoU = bb_intersection_over_union(
                                            (x[1:5].cpu().numpy() / yolo_frame_scale).astype(np.int),
                                            [c1[0], c1[1], c2[0], c2[1]], False)
                                        if IoU > 0.05:
                                            IoU_fake_condition = False
                                            break
                                if IoU_fake_condition and (mean_dst_buffer[i] > 7 or fake_dst_bufer[i] < 2.5):
                                    pre_fake_feature_vectors.extend(temp_fature_vectors[i])

                    if index != -1 and len(pre_feature_vectors) or signal_start and not lock_flag:
                        ad_confidence = True
                        if (min_dst_total < threshold or signal_start ) and (signal_counter > 3 or lock_flag or not start_funcion) and IoU_condition:
                            pre_feature_vectors.append(temp_fature_vectors[index])
                            if len(pre_feature_vectors) > 20:
                                pre_feature_vectors.pop(2)
                    elif index == -1:
                        ad_confidence = False
                    else:
                        ad_confidence = True


                    if (index != -1 or signal_start) and YOLO_count < 5:
                        YOLO_count += 1
                    elif YOLO_count > 0:
                            YOLO_count -= 1

                    if lock_flag:
                        YOLO_time += time.time() - start_time_temp
                else:
                    ad_confidence = False


            if signal_start:
                signal_counter += 1
            elif signal_counter > 0:
                signal_counter -= 0.5


            if signal_counter >= 5:
                lock_flag = True

            if lock_flag:
                if ad_confidence and threshold > 3:
                    threshold -= 0.01
                elif not ad_confidence and threshold < 4.5:
                    threshold += 0.01

            #threshold = 5

            if AD_bb is not None and YOLO:
                YOLO_tracking_found_counter += 1

            # Draw box and text
            if True or (not start_funcion or lock_flag):
                if (ad_confidence or YOLO_count > 0 or signal_start and not lock_flag) and AD_bb is not None:
                    if signal_start and not lock_flag:
                        color = (0,255,255)
                        text = "Initializing"
                    else:
                        if YOLO:
                            color = (0, 255, 255)
                            text = "Tracking: YOLO"
                        else:
                            color = (0, 255, 0)
                            text = "Tracking: ADNet"

                    found_count += 1
                    cv2.rectangle(ad_frame, (AD_bb[0], AD_bb[1]), (AD_bb[0]+AD_bb[2], AD_bb[1]+AD_bb[3]), color, 1)
                    c1 = (AD_bb[0], AD_bb[1])
                    c2 = (AD_bb[0] + AD_bb[2], AD_bb[1] + AD_bb[3])
                    ad_frame, signals = drawTracking(ad_frame, c1, c2, PoseNet)

                    if signals[4]:
                        if stop_counter < 3:
                            stop_counter += 1
                    elif stop_counter > 0:
                        stop_counter -= 0.2
                    else:
                        stop_counter = 0

                    if stop_counter >= 3:
                        lock_flag = False
                        pre_fake_feature_vectors = []
                        pre_feature_vectors = []
                        break

                    t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                    c1_tag = tuple((int((c1[0] + c2[0] - t_size[0]) / 2), c1[1] - (t_size[1] + 4)))
                    c2_tag = c1_tag[0] + t_size[0] + 3, c1_tag[1] + t_size[1] + 4
                    cv2.rectangle(ad_frame, c1_tag, c2_tag, color, -1)
                    cv2.putText(ad_frame, text, (c1_tag[0], c1_tag[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0],1);
                    count_n_detect = 0

                else:
                    count_n_detect += 1
                    found_count -= 1
                    c1 = [-1, -1]
                    c2 = [-1, -1]


                torch.cuda.empty_cache()

            # Clean up
            if len(pre_fake_feature_vectors) > 15:
                del pre_fake_feature_vectors[0:len(pre_fake_feature_vectors) - 15]

            pre_length = len(pre_fake_feature_vectors)

            if record:
                video.write(ad_frame)

            if save_bb:
                bb_file.write(str(c1[0])+","+str(c1[1])+","+str(c2[0])+","+str(c2[1])+","+str(int(not YOLO))+"\n")

            if rotations[video_index] == 2:
                ad_frame = cv2.resize(ad_frame, (int(800 * frame.shape[1] / frame.shape[0]), 1000))
            cv2.imshow("frame", ad_frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

        else:
            break

    stop_time = time.time()

    cap.release()
    if record:
        video.release()
    if save_bb:
        bb_file.close()

    cv2.destroyAllWindows()


    print("Yolo Trackings:", YOLO_tracking_counter)
    print("YOLO Found trackigs:", YOLO_tracking_found_counter)
    print("ADNet Trackings:", ADNet_tracking_counter)
    print("Re-detections:", redetect_counter)

    print("Frames per second:", (stop_time-start_time)/frame_counter)
    print("ADNet time:", ADNet_time/ADNet_tracking_counter)
    print("YOLO time:", YOLO_time/YOLO_tracking_counter)

    if save_stats:
        print(name)
        stats_file.write(name + "," + str(YOLO_tracking_counter) + "," + str(YOLO_tracking_found_counter) + "," + str(ADNet_tracking_counter) + "," + str(redetect_counter) + "," + str(total_track_counter ) + "," + str(frame_counter/(stop_time-start_time))+ "," + str(ADNet_tracking_counter/ADNet_time) + "," + str(YOLO_tracking_counter/YOLO_time) + "," + str(frame_counter) + "\n")
