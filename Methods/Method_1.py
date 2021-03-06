from __future__ import division
from YOLO.YOLO import *
torch.cuda.current_device()
import posenet
from SiameseNetwork.SiameseNetwork import *
from Extra_Functions import *
import time


# User Options
record = False
start_funcion = True
save_bb = False
save_stats = False


# YOLO setup
args = arg_parse()
YOLO, CUDA, batch_size, confidence, nms_thesh, num_classes, classes, inp_dim = init_YOLO(args)
YOLO.eval()

# ReID setup
id_path = r"C:/Users/Ma-Ly/Google Drev/DTU - Speciale F2020/Siamese_Networks/models/QaudrupletHardNet_Feb_26_2020_1242.pt"
id_model = load_model(id_path, 1)
id_model.eval()

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

# Test videos
samples = [0,
            "../Project_V2/Test_data/test_video.mp4",
           "../Project_V2/Test_data/1_person_easy.mp4",
           "../Project_V2/Test_data/1_person_difficult.mp4",
           "../Project_V2/Test_data/2_person_easy.mp4",
           "../Project_V2/Test_data/2_person_difficult.mp4",
           "../Project_V2/Test_data/20200518_100805.mp4",
           "../Project_V2/Test_data/20200519_214042.mp4",
           "../Project_V2/Test_data/20200519_214131.mp4",
           "../Project_V2/Test_data/20200519_214325.mp4",
           "../Project_V2/Test_data/20200519_214554.mp4",
           "../Project_V2/Test_data/20200519_214720.mp4",
           "../Project_V2/Test_data/20200519_214820.mp4",
           "../Project_V2/Test_Data/20200519_214847.mp4",
           "../Project_V2/Test_data/20200519_214910.mp4",
           "../Project_V2/Test_data/20200603_205814.mp4",
           "../Project_V2/Test_data/20200603_205901.mp4",
           "../Project_V2/Test_data/20200603_205942.mp4",
           "../Project_V2/Test_data/20200603_210036.mp4",
           "../Project_V2/Test_data/20200603_210109.mp4",
           "../Project_V2/Test_data/20200603_210228.mp4",
           "../Project_V2/Test_data/20200603_210319.mp4"]
#Video rotations
rotations = [0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,2,2,2,2,2,2,2]

#Loop through test videos
for video_index in range(5,6):
    if video_index == 2 or video_index == 4 or video_index == 10 or video_index == 11: #Videos where start signal is not detected
        continue
    video_name = samples[video_index]
    cap = cv2.VideoCapture(video_name)
    assert cap.isOpened(), 'Cannot capture source'

    # Initiliaze variables
    found_flag = False
    lock_flag = False
    true_feature_buffer = []
    false_feature_buffer = []
    signal_counter = 0
    count_found= 0
    image_threshold = 4.5
    frame_scale = 1
    yolo_frame_scale = 0.2
    stop_counter = 0
    IoU_condition = True
    pre_length = 0
    frame_counter = 0

    # Set up recorder
    if rotations[video_index] == 2:
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * frame_scale)
        img_width  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * frame_scale)
    else:
        img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*frame_scale)
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*frame_scale)
    fps = 20#int(cap.get(cv2.CAP_PROP_FPS))

    if not isinstance(video_name, int):
        timestamp = time.strftime('%b_%d_%Y_%H%M', time.localtime())
        name = os.path.splitext(os.path.basename(video_name))[0]

        if record:
            fourcc = cv2.VideoWriter_fourcc(*'FMP4')
            video = cv2.VideoWriter('Results/Method 1/' + name + "_"+ timestamp + '.mp4', fourcc, fps, (img_width, img_height))

    # Setup of bounding box file
    if save_bb:
        name = os.path.splitext(os.path.basename(video_name))[0]
        bb_file= open("Results/Method 1/bounding boxes/"+name+"_"+timestamp+".txt","w+")

    # Setup of statistics file
    if save_stats and not 'stats_file' in locals():
        name = os.path.splitext(os.path.basename(video_name))[0]
        stats_file= open("Results/Method 1/Statistics/stats_"+timestamp+".txt","w+")
        stats_file.write("Video name , Frames per second\n")

    start_time = time.time()


    # Start executing input
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_counter += 1
            if rotations[video_index] == 1:
                frame = cv2.flip(frame, -1)
            elif rotations[video_index] == 2:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Prepare frame and Run YOLO
            frame = cv2.resize(frame, (int(frame.shape[1] * frame_scale), int(frame.shape[0] * frame_scale)))
            yolo_frame = cv2.resize(frame, (int(frame.shape[1] * yolo_frame_scale), int(frame.shape[0] *  yolo_frame_scale)))
            img = prep_image(yolo_frame, inp_dim)
            im_dim = yolo_frame.shape[1], yolo_frame.shape[0]
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = YOLO(img.cuda(), CUDA)
            output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

            # If not object is found
            if type(output) == int:
                if found_flag:
                    found_flag = False

                if count_found< 0:
                    count_found= 0
                else:
                    count_found+= 1

                c1 = [-1, -1]
                c2 = [-1, -1]
            else:
                #Scale boxes to fit original image
                im_dim = im_dim.repeat(output.size(0), 1)
                scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)
                output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
                output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2
                output[:, 1:5] /= scaling_factor

                # Fit box sizes to image
                for i in range(output.shape[0]):
                    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

                # Init variables
                min_dst = 1000
                flag = False
                condition = False
                min_condition = False
                dst = 100


                index = 0
                min_index = 0
                fake_feature_vectors = list()
                mean_dst_buffer = []
                fake_dst_bufer = []
                min_dst_buffer = []

                # Loop over all bb
                for x in output:
                    dst_temp = 100
                    # If bb is too small ignore
                    if start_funcion and (not lock_flag or not start_funcion) and (abs(x[2]-x[4]) < img_height*0.1*yolo_frame_scale or abs(x[1]-x[3]) < img_width*0.05*yolo_frame_scale): # Set according to frame size
                        continue

                    # Set box size
                    c1_temp = tuple((x[1:3].cpu().numpy()/yolo_frame_scale).astype(np.int))
                    c2_temp = tuple((x[3:5].cpu().numpy()/yolo_frame_scale).astype(np.int))

                    # Prepare crop and run the Siamese network
                    id_frame = frame[c1_temp[1]:c2_temp[1], c1_temp[0]:c2_temp[0]]
                    id_frame = prepae_frame(id_frame).cuda()
                    with torch.no_grad():
                        feature_vector_temp = id_model(id_frame.unsqueeze(0))

                    # Test similarity from the last know imag
                    condition_temp, dst_temp, min_dst, fake_dst, mean_dst = test_similarity(feature_vector_temp, true_feature_buffer, false_feature_buffer, image_threshold)



                    # Check for user signals
                    if start_funcion and not lock_flag or count_found > 20 and min_dst < 4:
                        x_offset = int(abs(c1_temp[0] - c2_temp[0]) * 0.5)
                        # Offset bouding box to contain Start signal always
                        if c1_temp[0] - x_offset < 0:
                            x_offset = c1_temp[0]

                        if c2_temp[0] + x_offset >= frame.shape[1]:
                            x_offset = frame.shape[1] - c2_temp[0] - 1

                        profiles = PoseNet.check_signal(frame[int(c1_temp[1]):int(c2_temp[1]), int(c1_temp[0] - x_offset):int(c2_temp[0] + x_offset)], (c1_temp[1], c1_temp[0]-x_offset))
                        for profile in profiles:
                            if profile[0][1] or count_found > 20 and (profile[0][5] or profile[0][6]) and lock_flag:
                                profile_signal = profile
                                dst_temp = -1
                                condition_temp = True
                            if  (profile[0][5] or profile[0][6]):
                                false_feature_buffer = []


                    # Store results for later evaluations
                    fake_feature_vectors.append(feature_vector_temp)
                    mean_dst_buffer.append(mean_dst)
                    fake_dst_bufer.append(fake_dst)
                    min_dst_buffer.append(dst_temp)

                    #Set distance depending if the start signal was found
                    dst_temp = dst_temp if dst_temp == -1 else mean_dst

                    if dst > dst_temp and condition_temp:
                        min_index = index
                        flag = True
                        dst = dst_temp
                        feature_vector = feature_vector_temp
                        condition = condition_temp
                        c1 = c1_temp
                        c2 = c2_temp

                    index += 1

                #IoU Test - Update fake feature buffer
                if lock_flag:

                    if condition:
                        IoU_condition = True
                        # Check IoU with the true target
                        for x, i in zip(output, range(len(output))):
                            # If bb is too small ignore
                            if start_funcion and (not lock_flag or not start_funcion) and (abs(x[2]-x[4]) < img_height*0.2 or abs(x[1]-x[3]) < img_width*0.1) or i == min_index: # Set according to frame size
                                continue

                            # Measure IoU
                            IoU = bb_intersection_over_union((x[1:5].cpu().numpy()/yolo_frame_scale).astype(np.int),[c1[0],c1[1],c2[0],c2[1]],False)

                            # Handle bonding box and feature vector according to the IoU
                            if IoU > 0.05:
                                IoU_condition = False
                            if IoU > 0.05 and IoU < 0.4:
                                bb = (x[1:5].cpu().numpy()/yolo_frame_scale).astype(np.int).tolist()
                                newBB = findNewBox([c1[0],c1[1],c2[0],c2[1]], bb, False)

                                if newBB != bb:
                                    id_frame = frame[newBB[1]: newBB[3], newBB[0]: newBB[2]]
                                    id_frame = prepae_frame(id_frame).cuda()
                                    with torch.no_grad():
                                        feature_vector= id_model(id_frame.unsqueeze(0))

                                    false_feature_buffer.extend(feature_vector)

                            elif IoU <= 0.05:
                                false_feature_buffer.extend(fake_feature_vectors[i])
                    else:
                        #  Check IoU betwe all found bounding boxes
                        for x1, i in zip(output, range(len(output))):
                            IoU_fake_condition = True
                            for x2, j in zip(output, range(len(output))):
                                if i != j:
                                    IoU = bb_intersection_over_union((x[1:5].cpu().numpy() / yolo_frame_scale).astype(np.int), [c1[0], c1[1], c2[0], c2[1]], False)
                                    if IoU > 0.05:
                                        IoU_fake_condition = False
                                        break

                            if IoU_fake_condition and min_dst_buffer[i] > 1.5 and (mean_dst_buffer[i] >7 or  fake_dst_bufer[i] < 2.5):

                                false_feature_buffer.extend(fake_feature_vectors[i])

                # Update signal found counter
                if dst == -1:
                    if signal_counter < 10:
                        signal_counter += 1
                elif signal_counter > 1:
                    #signal_counter -= 2
                    signal_counter -= 0.2
                else:
                    signal_counter = 0

                # Set lock, so the tracking object can't change
                if signal_counter >= 5:
                    lock_flag = True

                # update found counter
                if condition:
                    if count_found> 0:
                        count_found= 0
                    else:
                        count_found-= 1

                    #Clean fake vectors
                    temp_fake = []
                    if count_found== - 3:
                        found_flag = True
                        for vec in false_feature_buffer:
                            dst = F.pairwise_distance(feature_vector, vec).item()
                            if dst > 1:
                                temp_fake.append(vec)

                        false_feature_buffer = temp_fake
                else:
                    if count_found < 0:
                        count_found = 0
                    else:
                        count_found += 1

                # Reset buffers
                if(count_found> 50 or dst == -1 and signal_counter == 3) and not lock_flag:
                    false_feature_buffer = []
                    true_feature_buffer = []


                # Update true feature buffer
                if IoU_condition and flag and ( condition and lock_flag or dst == -1  and signal_counter > 3):
                    true_feature_buffer.append(feature_vector)
                    if len(true_feature_buffer) > 20:
                        true_feature_buffer.pop(2) # To asure we always have the first to clear pics, which can be used to reset

                    if image_threshold > 3.5:
                        image_threshold -= 0.01
                elif image_threshold < 4.5:
                    image_threshold += 0.001


                # Draw box and text
                if (not start_funcion or lock_flag) and flag:
                    frame, signals = drawTracking(frame, c1, c2, PoseNet)

                    # Check for stop signal
                    if signals[4]:
                        if stop_counter < 3:
                            stop_counter += 1
                    elif stop_counter > 0:
                        stop_counter -= 0.2
                    else:
                        stop_counter = 0

                    if stop_counter >= 3:
                        lock_flag = False
                        false_feature_buffer = []
                        true_feature_buffer = []
                        image_threshold = 4.5
                        break
                else:
                    # Target user not found
                    c1 = [-1,-1]
                    c2 = [-1,-1]
                    if stop_counter > 0:
                        stop_counter -= 0.2
                    else:
                        stop_counter = 0

                # Clean up
                if len(false_feature_buffer) > 15:
                    del false_feature_buffer[0:len(false_feature_buffer)-15]
                elif len(false_feature_buffer) > 0 and len(false_feature_buffer) == pre_length:
                    del false_feature_buffer[0]

                del fake_feature_vectors
                torch.cuda.empty_cache()

                pre_length = len(false_feature_buffer)

            # Save frame
            if record:
                video.write(frame)

            # Save bounding box
            if save_bb:
                bb_file.write(str(c1[0])+","+str(c1[1])+","+str(c2[0])+","+str(c2[1])+"\n")

            # Show frame
            if rotations[video_index] == 2:
                frame = cv2.resize(frame,(int(800*frame.shape[1]/frame.shape[0]),1000))
            cv2.namedWindow("frajkme");
            cv2.imshow("frajkme", frame)
            key = cv2.waitKey(1)

            # Exit if Q-key pressed
            if key & 0xFF == ord('q'):
                break

        else:
            break

    # Close files
    cap.release()
    if record:
        video.release()
    if save_bb:
        bb_file.close()
    if save_stats:
        stats_file.write(name + "," + str(frame_counter/(time.time()-start_time)) + "\n")

    cv2.destroyAllWindows()



