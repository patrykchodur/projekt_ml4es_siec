import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.models import mobilenet_v2

from PIL import Image
import numpy as np
import glob

chinatree = []
fig = []
judastree = []
palm = []
pine = []
labels_no = int(2)

def load_images(directory):
    result = []
    for f in glob.iglob(directory + "/*"):
        img = np.asarray(Image.open(f))
        if not img.shape == (224, 224, 3):
            # raise ValueError("Wrong file size: " + str(img.shape) + ", name: " + str(f))
            continue
        img = np.float32(img/255)
        result.append(img)
    result = np.stack(result, axis=0)
    result = np.moveaxis(result, 3, 1)
    return result

def labels(images, label):
    result = np.zeros(images.shape[0], dtype=int) + label
    return result

def labels_matrix(labels_list, labels_no):
    result = np.zeros((labels_no, len(labels_list)))
    for i in range(len(labels_list)):
        result[labels_list[i]][i] = 1
    return result

chinatree_images = load_images("test_cut")
chinatree_labels = labels(chinatree_images, 0)

palm_images = load_images("test2_cut")
palm_labels = labels(palm_images, 1)

all_images = np.concatenate((chinatree_images, palm_images), 0)
all_labels = np.concatenate((chinatree_labels, palm_labels), 0)

shuffler = np.random.permutation(all_images.shape[0])
all_images_shuffled = all_images[shuffler]
print(all_images_shuffled.shape)
# all_labels_shuffled = labels_matrix(all_labels[shuffler], labels_no).T
all_labels_shuffled = all_labels[shuffler]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(44944, 84)
        self.fc2 = nn.Linear(84, 62)
        self.fc3 = nn.Linear(62, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 44944)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=-1)

        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    inputs = torch.tensor(all_images_shuffled)
    labels = torch.tensor(all_labels_shuffled)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    print(inputs.shape, all_labels_shuffled.shape, outputs.shape)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()

    print('[%d] loss: %.3f' %
            (epoch + 1, running_loss / 2000))
    running_loss = 0.0
"""
    for i in range(all_images_shuffled.shape[0]):
        # get the inputs; data is a list of [inputs, labels]
        inputs = torch.tensor(np.expand_dims(all_images_shuffled[i], 0))
        labels = all_labels_shuffled[i]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        print(inputs.shape, all_labels_shuffled.shape, outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
"""

print('Finished Training')


# saving

#model = mobilenet_v2(pretrained=True)

model.eval()
#input_tensor = torch.rand(1,3,224,224)
#
#script_model = torch.jit.trace(model,input_tensor)


script_model.save("my_model.pt")

