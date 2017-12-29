from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F



fc1Width = 120*25

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 60, kernel_size=5 )
        self.conv2 = nn.Conv2d( 60, 120, kernel_size=5 )
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear( fc1Width, 800 )
        self.fc2 = nn.Linear( 800, 395)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #  import ipdb; ipdb.set_trace()
        x = x.view(-1, fc1Width )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1 )





