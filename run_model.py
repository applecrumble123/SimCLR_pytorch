import torch
import os
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


from typing import Optional, Callable, Tuple, Any
from PIL import Image
import torchvision.models as models

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import config


device = torch.device("cuda:{}".format(config.GPU_NUM) if torch.cuda.is_available() else "cpu")

class_labels = []
class_labels_num = []

count = 0
for folder in os.listdir(config.TRAIN_DIR):
  if folder.startswith('.') is False:
    class_labels.append(folder)
    class_labels_num.append(count)
    count = count + 1

print(class_labels)
print(class_labels_num)

training_image_names = []
training_class_labels = []

for i in range(len(class_labels)):

  # path of images for each of the class
  sub_path = config.TRAIN_DIR + class_labels[i] + '/'
  #print("sub_path: \n", sub_path, "\n")

  # get the images in each class
  sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
  #print("sub_file_names: \n", sub_file_names, "\n")

  # get the list of training images
  training_image_names = training_image_names + sub_file_names
  #print("training_file_names: \n", training_file_names, "\n")

  # get the total number of class labels per image folder,
  sub_class_labels = [i] * len(sub_file_names)
  #print("sub_food_labels: \n", sub_food_labels, "\n")

  # get the total number of training labels
  training_class_labels = training_class_labels + sub_class_labels
  #print("training_food_labels: \n", training_food_labels, "\n")

test_image_names = []
test_class_labels = []

for i in range(len(class_labels)):

  # path of images for each of the class
  sub_path = config.TEST_DIR + class_labels[i] + '/'
  #print("sub_path: \n", sub_path, "\n")

  # get the images in each class
  sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
  #print("sub_file_names: \n", sub_file_names, "\n")

  # get the list of training images
  test_image_names = test_image_names + sub_file_names
  #print("training_file_names: \n", training_file_names, "\n")

  # get the total number of class labels per image folder,
  sub_class_labels = [i] * len(sub_file_names)
  #print("sub_food_labels: \n", sub_food_labels, "\n")

  # get the total number of training labels
  test_class_labels = test_class_labels + sub_class_labels
  #print("training_food_labels: \n", training_food_labels, "\n")

class ImageDataset(Dataset):

    def __init__(self, class_labels, images, transform: Optional[Callable] = None,):
        super().__init__()
        self.transform = transform
        self.class_labels = class_labels
        self.images = images

    def __len__(self):
        return len(self.class_labels)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.images[index], self.class_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        if self.transform is None:
            # convert PIL to tensor
            img = transforms.ToTensor()(img)
        return img, target

# projection layer at the end of the model
# map from the output of the encoder into the dimension that we want to have for the comparison
class ResNetSimCLR(nn.Module):

    def __init__(self):
        super(ResNetSimCLR, self).__init__()
        resnet = models.resnet50(pretrained=False)

        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # original resnet50 --> kernel_size=2, stride=2
        resnet.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        # 512
        #num_ftrs = resnet.fc.in_features
        # AdaptiveAvgPool2d(output_size=(1, 1))
        # get the features before the linear layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP
        #self.l1 = nn.Linear(num_ftrs, num_ftrs)
        #self.l2 = nn.Linear(num_ftrs, out_dim)

        self.l1 = nn.Linear(2048, 2048)
        self.l2 = nn.Linear(2048, 128)


    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        # Projection --> (Dense --> Relu --> Dense)
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x


if __name__ == '__main__':

    checkpoints_folder = config.MODEL_CHECKPOINT_FOLDER

    model = ResNetSimCLR()
    model.eval()

    state_dict = torch.load(config.SAVED_MODEL_PATH, map_location=torch.device('cpu'))

    model.load_state_dict(state_dict)
    model = model.to(device)

    """
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    """

    # remove the projection head in the model
    new_model = nn.Sequential(*list(model.children())[:-2])
    """
    for layer in new_model.children():
        print(layer)
    """
    # freeze all the layers in the resnet model
    for param in new_model.parameters():
        param.requires_grad = False
        # print(param)

    new_train_transformed_dataset = ImageDataset(class_labels=training_class_labels, images=training_image_names,
                                                 transform=None)
    new_train_dataloader = DataLoader(new_train_transformed_dataset, batch_size=256,
                                      shuffle=True, num_workers=3)

    new_test_transformed_dataset = ImageDataset(class_labels=test_class_labels, images=test_image_names,
                                                transform=None)

    new_test_dataloader = DataLoader(new_test_transformed_dataset, batch_size=256,
                                     shuffle=True, num_workers=3)

    """
    sample_batched[1] = class of images

    # Number of images
    len(sample_batched[0]) = 256 (number of images per batch)
    sample_batched[0] = images in tensor

    Size of images
    len(sample_batched[0][0] = 3
    len(sample_batched[0][0][0] = 32
    len(sample_batched[0][0][0][0] = 32

    """

    x_train = []
    y_train = []

    for i_batch, sample_batched in enumerate(new_train_dataloader):
        img = sample_batched[0].to(device)
        outputs = new_model(img).to(device)

        x_train.extend(outputs.cpu().detach().numpy())
        y_train.extend(sample_batched[1].cpu().numpy())

    X_train_feature = np.array(x_train)
    print("Train features")
    print(X_train_feature.shape)
    X_train = X_train_feature.reshape(len(x_train), len(x_train[0]) * len(x_train[0][0]) * len(x_train[0][0][0]))
    print(X_train.shape)

    x_test = []
    y_test = []

    for i_batch, sample_batched in enumerate(new_test_dataloader):
        img = sample_batched[0].to(device)
        outputs = new_model(img).to(device)

        x_test.extend(outputs.cpu().detach().numpy())

        for j in sample_batched[1]:
            np_j = j.cpu().numpy()
            y_test.append(np_j)

    X_test_feature = np.array(x_test)

    print("Test features")
    print(X_test_feature.shape)

    X_test = X_test_feature.reshape(len(x_test), len(x_test[0]) * len(x_test[0][0]) * len(x_test[0][0][0]))

    print(X_test.shape)

    logistic_regression = LogisticRegression(random_state=0, max_iter=5000, solver='lbfgs', C=1.0)
    logistic_regression.fit(X_train, y_train)

    y_predict = logistic_regression.predict(X_test)
    print(y_predict)

    acc = accuracy_score(y_test, y_predict)
    print("Accuracy is {}%".format(acc * 100))