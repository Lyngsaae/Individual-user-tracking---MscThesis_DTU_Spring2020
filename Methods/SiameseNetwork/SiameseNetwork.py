import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F


class TripletNetwork(nn.Module):
    def __init__(self):
        super(TripletNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 200 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # return output1, output2
        output3 = self.forward_once(input3)
        return output1, output2, output3

class TripletNetwork_v2(nn.Module):
    def __init__(self):
        super(TripletNetwork_v2, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            #old version
           #nn.BatchNorm2d(3),
           #nn.ReflectionPad2d(1),
           #nn.Conv2d(3, 4, kernel_size=3),
           #nn.ReLU(inplace=True),
           #nn.BatchNorm2d(4),

           #nn.ReflectionPad2d(1),
           #nn.Conv2d(4, 8, kernel_size=3),
           #nn.ReLU(inplace=True),
           #nn.BatchNorm2d(8),

           #nn.MaxPool2d(2, 2),
           #nn.ReflectionPad2d(1),
           #nn.Conv2d(8, 16, kernel_size=3),
           #nn.ReLU(inplace=True),
           #nn.BatchNorm2d(16),

           #nn.ReflectionPad2d(1),
           #nn.Conv2d(16, 8, kernel_size=3),
           #nn.ReLU(inplace=True),
           #nn.BatchNorm2d(8),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 50, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 10))

    def forward(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

class QdrupletNetwork(nn.Module):
    def __init__(self):
        super(QdrupletNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 50, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 10))

    def forward(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 200 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 50))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class SiameseNetwork_Classification(nn.Module):
    def __init__(self):
        super(SiameseNetwork_Classification, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 200 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 100),
            nn.ReLU(inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(100 * 2, 500),
            nn.ReLU(),
            nn.Linear(500, 2),
            nn.Softmax(dim=1))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = torch.cat((output1, output2), 1)
        result = self.fc2(output3)
        return result

class TripletMetricNetwork(nn.Module):
    def __init__(self):
        super(TripletMetricNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 200 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 100),
            nn.ReLU(inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(100 * 2, 500),
            nn.ReLU(),
            nn.Linear(500, 1))
        # nn.Softmax(dim=1))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward_once_metric(self, x1, x2):
        cat = torch.cat((x1, x2), 1)
        result = self.fc2(cat)
        return result

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        cat_1 = torch.cat((output1, output2), 1)
        cat_2 = torch.cat((output1, output3), 1)
        result1 = self.fc2(cat_1)
        result2 = self.fc2(cat_2)
        return result1, result2

def prepae_frame(frame):
    transform = transforms.Compose([transforms.Resize((200, 100))
                                    #,transforms.Grayscale()
                                    ,transforms.ToTensor()
                                    #,transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])
    frame = Image.fromarray(frame)
    frame = transform(frame)

    return frame

def load_model(path, siamese_net = 0):
    model = torch.load(path).cuda()
    model.eval()

    return model

def test_similarity(feature_vector, pre_feature_vectors, fake_feature_vectors, threshold):
    # Calculate distances
    dst = [F.pairwise_distance(feature_vector,  vec).item() for vec in pre_feature_vectors]
    dst_fake = [F.pairwise_distance(feature_vector, vec).item() for vec in fake_feature_vectors]

    dst.sort()
    if len(pre_feature_vectors) > 0:
        min_dst = min(dst)
        total_mean_dst = sum(dst) / len(dst)
        mean_dst = sum(dst[0:3]) / 3
        if mean_dst == 0:
            mean_dst = 0.1
    else:
        min_dst = 1
        mean_dst = 1
        total_mean_dst = 1

    # Check if the the fake one has been found multiple times
    dst_fake.sort()
    if len(dst_fake) > 10 and False:
        min_fake_dst = sum(dst_fake[0:3])/3
    elif len(dst_fake) > 0:
        min_fake_dst = dst_fake[0]
    else:
        min_fake_dst = 10000

    if min_fake_dst < 2 and min_dst > 1.5:
        condition = False
    elif min_dst < 1.5 and min_fake_dst > min_dst:
        condition = True
    elif total_mean_dst < threshold and min_fake_dst > min_dst:
        condition = True
    else:
        condition = False

    #print(min_dst, min_fake_dst, total_mean_dst, threshold)
    #print("Min distance:", min_dst)
    #print("Min fake distance:", min_fake_dst)
    #print("Mean distance:", mean_dst, " - Condition:", mean_dst < threshold and min_dst < min_fake_dst)
    #return ((mean_dst < threshold or min_dst < 1) and min_dst < min_fake_dst), mean_dst, min_dst, min_fake_dst, 1
    return condition, mean_dst, min_dst, min_fake_dst, total_mean_dst

def test_similarity_v2(img, vecs, model, threshold):
    output = model.forward_once(img.unsqueeze(0))
    min_dst = 1000

    print("Min distance:", min_dst)
    if len(vecs) > 0:
        total_vec = vecs[0]
        for vec in vecs[1:]:
            total_vec +=vec

        total_vec /= len(vecs)

        print(total_vec.shape)
        min_dst = F.pairwise_distance(output,  total_vec).item()
        print("Distance:", min_dst)

    print("Min distance:", min_dst)
    return min_dst < threshold, min_dst, output

def test_similarity_v3(img, vecs, model, threshold):

    min_dst = 1000
    dst = -1
    for vec in vecs:#[-6:-1]:
        dst += model(img.unsqueeze(0), vec.unsqueeze(0))[0,1].item()

    if dst != -1:
        min_dst = dst/len(vecs)
    print("Min distance:", min_dst)
    return min_dst < threshold, min_dst, 0

def make_sim_table(model, img, true_vecs, fake_vecs):
    output = model.forward_once(img.unsqueeze(0))
    score = np.zeros((1+len(fake_vecs),1))
    score[0,0], min_dst = test_similarity_v4(output, true_vecs)

    for vecs, index in zip(fake_vecs,range(1,len(fake_vecs)+1)):
        score[index,0], min_dst = test_similarity_v4(output, vecs)

    return np.array(score,)

def test_similarity_v4(output, vecs):
    min_dst = 1000
    mean_dst = 0
    index = 0

    for vec in vecs:
        dst = F.pairwise_distance(output,  vec).item()
        mean_dst +=dst
        if dst < min_dst:
            min_dst = dst
        index += 1

    if len(vecs) > 0:
        mean_dst /= len(vecs)
    else:
        mean_dst = 1
        min_dst = 1

    return mean_dst, min_dst