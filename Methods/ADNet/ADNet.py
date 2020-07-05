import numpy as np
import torch.nn as nn
import torch
import cv2
import scipy.io as sio

class ADNet(nn.Module):
    def __init__(self, m, k):
        super(ADNet, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(0.05),

            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.05),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.05),

            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.05),

            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.05),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 50 * 25, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512 + m * k, m + 2)
        )

        self.softmax = nn.Softmax(-1)
        self.m = m

    def forward(self, x, pre_actions):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        output = torch.cat((output, pre_actions), 1)
        output = self.fc2(output)

        action = self.softmax(output[:, 0:self.m])
        confidence = self.softmax(output[:, self.m:])

        return action, confidence
class ADDuelingNet(nn.Module):
    def __init__(self, m, k):
        super(ADDuelingNet, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.fc_v = nn.Sequential(
            nn.Linear(8 * 50 * 25 + m * k, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2)
        )

        self.fc_a = nn.Sequential(
            nn.Linear(8 * 50 * 25 + m * k, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, m)
        )

        self.softmax = nn.Softmax(-1)
        self.m = m

    def forward(self, x, pre_actions):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = torch.cat((output, pre_actions), 1)
        v = self.fc_v(output)
        a = self.fc_a(output)
        q = v[:, 0:1] + a - a.mean()

        return self.softmax(q), self.softmax(v)[:,0]

class ADNet_ORI(nn.Module):
    def __init__(self, m, k):
        super(ADNet_ORI, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, padding=0,stride= 2)
        self.conv2 = nn.Conv2d(3, 96, kernel_size=7, padding=0, stride=2)
        self.conv3 = nn.Conv2d(3, 96, kernel_size=7, padding=0, stride=1)
        self.ReLU = nn.ReLU(inplace=True)
        self.MaxPool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Conv2d(3, 96, kernel_size=7, padding=0, stride=1)
        self.fc2 = nn.Linear(512, 51)
        self.fc3 = nn.Linear(512, 51)
        self.fc4 = nn.Linear(512, 51)
        self.Softmax = nn.Softmax(-1)

    def forward(self, x, pre_actions):
        output = self.MaxPool(self.ReLU(self.conv1(x)))
        output = self.MaxPool(self.ReLU(self.conv2(output)))
        output = self.ReLU(self.conv3(output))
        output = self.ReLU(self.fc1(output))
        output = output.view(output.size(0), -1)

        action = self.Softmax(self.fc3(output))
        confidence = self.Softmax(self.fc4(output))

        return action, confidence

def init_ADNet_ORI():
    param_path = r"C:\Users\Ma-Ly\OneDrive\DTU\Elektroteknologi Kandidat\4. semester\Speciale\Code\Research\Action-Decision\ADNet-tensorflow-master\ADNet-tensorflow-master\ADNet_params.mat"
    initial_params = sio.loadmat(param_path)
    AD_path = r"C:/Users/Ma-Ly/Google Drev/DTU - Speciale F2020/Action_Decision_Network/Models/ADNet_SL_Advanced_Mar_23_2020_1050.pt"#ADNet_SL_Mar_19_2020_1215.pt"
    #initial_params = torch.load(AD_path)
    net = ADNet_ORI(10,11)
    for layer in initial_params:
        try:
            print(layer, ":", initial_params[layer].shape)
        except:
            print(layer)
            pass


    for name, param in net.named_parameters():
        name_def = name.split('.')
        print(name_def[0], "Before:", param.data.shape)
        if len(param.data.shape) == 4:
            print("perm")
            param.data = (torch.tensor(initial_params[name_def[0] + ('w' if name_def[1] == 'weight' else 'b')]).permute(3,2,1,0))
        elif len((torch.tensor(initial_params[name_def[0] + ('w' if name_def[1] == 'weight' else 'b')])).shape) == 4:
            param.data = (torch.tensor(initial_params[name_def[0] + ('w' if name_def[1] == 'weight' else 'b')])).squeeze(0).squeeze(0).permute(1,0)
        else:
            param.data = (torch.tensor(initial_params[name_def[0] + ('w' if name_def[1] == 'weight' else 'b')])).squeeze(0)

    return net

class ADagent():
    def __init__(self, model, transform, action_type = 1):
        self.model = model
        self.transform = transform
        self.action_reverse = tuple((1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 10)) if action_type == 1 else tuple((2,3,0,1,6,7,4,5, 10,11,8,9,14,15,12,13,16))
        self.action_type = action_type
        self.reset()
        self. label_names = tuple(('Left-Left', 'Double Left-Left', 'Left-Right', 'Double Left-Right', 'Right-Right', 'Double Right-Right', 'Right-Left', 'Double Right-Left', 'Top-Up', 'Double Top-Up', 'Top-Down', 'Double Top-Down', 'Bottom-Down', 'Doube Bottom-Down', 'Bottom-Up', 'Double Bottom-Up', 'Stop'))

    def reset(self):
        self.confidence_count = 0
        self.bb = None
        pre_action = [0] * len(self.action_reverse) * 10
        pre_action = [float(i) for i in pre_action]
        self.pre_actions_init = torch.tensor([pre_action], requires_grad=True).cuda()
        self.pre_actions = self.pre_actions_init.clone()

    def takeAction(self, img, bb = None):
        action_acounter = 0
        actions_performed = np.zeros((len(self.action_reverse)), dtype= np.int)
        action = -1
        if bb is not None:
            self.bb = np.array(bb)

        if self.bb is None:
            return None, False

        #print("Start bb:", self.bb)
        while True:
            height = float(self.bb[3]) * 0.03
            width = float(self.bb[2]) * 0.03
            width = width if width >= 1 else 1
            height = height if height >= 1 else 1

            if self.action_type == 1:
                actions = np.array(([-width, 0, 0, 0],  # left
                                    [width, 0, 0, 0],  # right
                                    [0, -height, 0, 0],  # Up
                                    [0, height, 0, 0],  # Down
                                    [-width * 3, 0, 0, 0],  # Double Left
                                    [width * 3, 0, 0, 0],  # Double Right
                                    [0, -height * 3, 0, 0],  # Double up
                                    [0, height * 3, 0, 0],  # Double down
                                    [width / 2, height / 2, -width, -height],  # Scale up # Need to get fixed
                                    [-width / 2, -height / 2, width, height, ],  # Scale down # Need to get fixed
                                    [0, 0, 0, 0]), dtype=np.int)  # None
            else:
                actions = np.array(([-width, 0, width, 0],  # left
                                    [-width * 3, 0, width * 3, 0],  # left
                                    [width, 0, -width, 0],  # right
                                    [width * 3, 0, -width * 3, 0],  # right
                                    [0, 0, width, 0],  # left
                                    [0, 0, width * 3, 0],  # left
                                    [0, 0, -width, 0],  # right
                                    [0, 0, -width * 3, 0],  # right
                                    [0, -height, 0, height],  # Up
                                    [0, -height * 3, 0, height * 3],  # Up
                                    [0, height, 0, -height],  # Up
                                    [0, height * 3, 0, -height * 3],  # Up
                                    [0, 0, 0, height],  # Up
                                    [0, 0, 0, height * 3],  # Up
                                    [0, 0, 0, -height],  # Up
                                    [0, 0, 0, -height * 3],  # Up
                                    [0, 0, 0, 0]), dtype=np.int)  # None

            self.bb[self.bb< 1] = 1
            if self.bb[1]+self.bb[3] > img.shape[0]:
              self.bb[1] = img.shape[0]-self.bb[3]
            if self.bb[0]+self.bb[2] > img.shape[1]:
              self.bb[0] = img.shape[1]-self.bb[2]

            img_crop = img[self.bb[1]:self.bb[1]+self.bb[3],self.bb[0]:self.bb[0]+self.bb[2],:]

            img_crop = self.transform(img_crop)
            #img_crop = (img_crop - 0.5) * 2
            with torch.no_grad():
                action_temp, confidence = self.model(img_crop.unsqueeze(0).cuda(), self.pre_actions)
            action_temp = action_temp.max(1)[1].view(1, 1)
            if action != self.action_reverse[action_temp]:
                action = action_temp
            else:
                action = len(actions)-1

            a = confidence[0,0].item()

            actions_performed[action] += 1
            action_encoded = torch.tensor(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)).unsqueeze(0).cuda() if self.action_type == 1 else torch.tensor(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.float32)).unsqueeze(0).cuda()
            self.pre_actions = torch.cat((action_encoded, self.pre_actions[:,:-len(actions)].float()), 1)
            self.pre_actions[:, action] = 1
            #print(confidence[0,0].item())
            if confidence[0,0].item() > 0.4:
                confidence = True
                self.confidence_count = 0
            elif False and self.confidence_count < 5:
                confidence = True
                self.confidence_count += 1
            else:
                confidence = False
                self.confidence_count = 0

            if action_acounter > 20 or action == len(actions)-1 or not confidence:
                self.pre_actions = self.pre_actions_init.clone()
                break



            #print(self.bb[0].item(),self.bb[1].item(),self.bb[2].item(),self.bb[3].item(), "+")
            self.bb += actions[action]
            #print( actions[action], "\n=", self.bb[0].item(), self.bb[1].item(), self.bb[2].item(), self.bb[3].item())
            self.bb[self.bb < 1] = 1
            if self.bb[1] + self.bb[3] > img.shape[0]:
                self.bb[3] = img.shape[0] - self.bb[1]-1
            if self.bb[0] + self.bb[2] > img.shape[1]:
                self.bb[2] = img.shape[1] - self.bb[0] - 1
            action_acounter += 1

        #self.bb[3] = torch.tensor(int(1.1 *float(self.bb[3].item())))
        #if self.bb[1] + self.bb[3] > img.shape[0]:
        #     self.bb[3] = img.shape[0] - self.bb[1] - 1

        #print("Confidence:", confidence, "\t - \t Total actions:", action_acounter, "\t - \t Actions perfromed:",actions_performed)
        return self.bb, confidence
