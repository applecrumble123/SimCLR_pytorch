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


"""
## create training and validation split
split = int(0.8 * len(training_class_labels))
index_list = list(range(len(training_class_labels)))
train_idx, valid_idx = index_list[:split], index_list[split:]
"""

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



#--------- Gaussian Blur Technique ---------------
class GaussianBlur(object):
    # Implements Gaussian Blur as described in the SimCLR paper
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):

        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample



#--------- Implement Data Transformations ----------------
class SimCLRTrainDataTransforms(object):

    # this function returns none
    # the -> None just tells that f() returns None (but it doesn't force the function to return None)
    def __init__(self,
                 # image size
                 input_height: int = 224,

                 # using gaussian blur for ImageNet, may not need for other dataset
                 gaussian_blur: bool = config.GAUSSIAN_BLUR_TRAIN,

                 # using jitter for ImageNet, may not need for other dataset
                 jitter_strength: float = config.COLOUR_DISTORTION_STRENGTH,

                 # normalization
                 normalize: Optional[transforms.Normalize] = None) -> None:

        # track the variables internally
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.normalize = normalize

        # jitter transform
        # from SimCLR paper
        # apply to only PIL images
        self.colour_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        # can apply to the same image and will get a different result every single time
        # apply to only PIL images
        data_transforms = [
            transforms.RandomResizedCrop(self.input_height),
            # p refers to the probability
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.colour_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ]

        # apply to only PIL images
        if self.gaussian_blur:
            data_transforms.append(GaussianBlur(kernel_size=int(0.1 * self.input_height), p=0.5))


        # normalization
        if self.normalize:
            data_transforms.append(normalize)

        # apply all the transformations to tensors when working in pytorch
        data_transforms.append(transforms.ToTensor())

        # transforms.Compose just clubs all the transforms provided to it.
        # So, all the transforms in the transforms.Compose are applied to the input one by one.
        self.train_transform = transforms.Compose(data_transforms)

    #  The __call__ method enables Python programmers to write classes where the instances behave like functions and can be called like a function
    # sample refers to an input image
    def __call__(self, sample):

        # call the instance self.train_transform and make it behave like a function
        # use the train_transform as specified in the initialization
        transform = self.train_transform

        # apply the transformation twice to 2 version of the same image as specified in the simclr paper
        xi = transform(sample)  # first version
        xj = transform(sample)  # second version

        return xi,xj




# -------- evaluation of the the data transformation ---------
class SimCLREvalDataTransform(object):
    # track these parameters internally
    def __init__(self,
                 input_height: int = 224,
                 normalize: Optional[transforms.Normalize] = None
                 ):

        self.input_height = input_height
        self.normalize = normalize

        # convert to tensor
        eval_data_transforms = [
            transforms.Resize(self.input_height),
            transforms.ToTensor()
        ]

        if self.normalize:
            eval_data_transforms.append(normalize)

        self.test_transforms = transforms.Compose(eval_data_transforms)

    # take the same input image used in the training, "sample"
    def __call__(self, sample):

        # call the instance self.test_transforms and make it behave like a function
        transform = self.test_transforms

        xi = transform(sample)  # first version
        xj = transform(sample)  # second version

        return xi, xj


# Loss function
# normalized temperature-scaled cross entropy loss
# output 1 and output 2 is the 2 different versions of the same input image
def nt_xent_loss(output1, output2, temperature):
    # concatenate v1 img and v2 img via the rows, stacking vertically
    out = torch.cat([output1, output2], dim=0)
    n_samples = len(out)

    # Full similarity matrix
    # torch.mm --> matrix multiplication for tensors
    # when a transposed is done on a tensor, PyTorch doesn't generate new tensor with new layout,
    # it just modifies meta information in Tensor object so the offset and stride are for the new shape --> its memory
    # layout is different than a tensor of same shape made from scratch
    # contiguous --> makes a copy of tensor so the order of elements would be same as if tensor of same shape created from scratch
    # --> https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107
    # the diagonal of the matrix is the square of each vector element in the out vector, which shows the similarity between the same elements
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov/temperature)

    # Negative similarity
    # creates a 2-D tensor with True on the diagonal for the size of n_samples and False elsewhere
    mask = ~torch.eye(n_samples, device=sim.device).bool()
    # Returns a new 1-D tensor which indexes the input tensor (sim) according to the boolean mask (mask) which is a BoolTensor.
    # returns a tensor with 1 row and n columns and sum it with the last dimension
    neg = sim.masked_select(mask).view(n_samples,-1).sum(dim=-1)

    # Positive similarity
    # exp --> exponential of the sum of the last dimension after output1 * output2 divided by the temp
    pos = torch.exp(torch.sum(output1 * output2, dim=-1)/temperature)
    # concatenate via the rows, stacking vertically
    pos = torch.cat([pos,pos], dim=0)

    # 2 copies of the numerator as the loss is symmetric but the denominator is 2 different values --> 1 for x, 1 for y
    # the loss will be a scalar value
    loss = -torch.log(pos/neg).mean()
    return loss

#----------- Implement projection head ----------------------

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

        self.l1 = nn.Linear(config.PROJECTION_LINEAR_1_INPUT, config.PROJECTION_LINEAR_1_OUTPUT)
        self.l2 = nn.Linear(config.PROJECTION_LINEAR_2_INPUT, config.PROJECTION_LINEAR_2_OUTPUT)


    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        # Projection --> (Dense --> Relu --> Dense)
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x



train_transformed_dataset = ImageDataset(class_labels = training_class_labels, images=training_image_names, transform=SimCLRTrainDataTransforms(input_height=config.TRAIN_IMG_INPUT_HEIGHT))
train_dataloader = DataLoader(train_transformed_dataset, batch_size=config.BATCH_SIZE,
                        shuffle=True, num_workers=3)

test_transformed_dataset = ImageDataset(class_labels = test_class_labels, images=test_image_names, transform=SimCLREvalDataTransform(input_height=config.EVAL_IMG_INPUT_HEIGHT))
test_dataloader = DataLoader(test_transformed_dataset, batch_size=config.BATCH_SIZE,
                        shuffle=True, num_workers=3)

"""
len(sample_batched) = 2 (returned images (2) + class)
len(sample_batched[0]) = 2 (number of augmented image version 1 and version 2 per batch)
len(sample_batched[0][0]) = 256 (image version 1 per batch)
len(sample_batched[0][1]) = 256 (image version 2 per batch)

# size of images
sample_batched[0][0][0] = 3
sample_batched[0][0][0][0] = 32
sample_batched[0][0][0][0][0] = 32
for i_batch, sample_batched in enumerate(train_dataloader):
    #print(i_batch)
    print(len(sample_batched[0][0][0][0][0]))
    break
"""

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)



class SimCLR(object):
    def __init__(self):

        self.writer = SummaryWriter()
        self.device = self._get_device()
        # predefined above
        self.nt_xent_loss = nt_xent_loss
        self.encoder = models.resnet50(pretrained=False)


    def _get_device(self):
        
        device = torch.device("cuda:{}".format(config.GPU_NUM) if torch.cuda.is_available() else "cpu")

        return device


    def _step(self, model, xis, xjs, n_iter):

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = nt_xent_loss(zis, zjs, temperature=config.NT_XENT_LOSS_TEMP)

        return loss



    def train(self):

        def get_mean_of_list(L):

            return sum(L) / len(L)


        model = ResNetSimCLR().to(self.device)

        model = self._load_pre_trained_weights(model)


        optimizer = torch.optim.SGD(model.parameters(), lr=config.OPTIMIZER_LR, weight_decay=config.OPTIMZER_WEIGHT_DECAY)

        #optimizer_LARS = LARSWrapper(optimizer, eta=1e-3)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(training_image_names), eta_min=0,
                                                               last_epoch=-1)


        model_checkpoints_folder = config.MODEL_CHECKPOINT_FOLDER

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_mean_batch_loss = np.inf

        for epoch_counter in range(config.EPOCHS):

            # a list to store losses for each epoch
            epoch_losses_train = []

            for i_batch, sample_batched in enumerate(train_dataloader):
                optimizer.zero_grad()

                xis = sample_batched[0][0].to(self.device)
                xjs = sample_batched[0][1].to(self.device)

                loss = self._step(model, xis, xjs, n_iter)

                # put that loss value in the epoch losses list
                epoch_losses_train.append(loss.to(self.device).data.item())

                # log every 50 steps
                if n_iter % 50 == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                loss.backward()

                optimizer.step()
                n_iter += 1
            #print("Epoch:{}".format(epoch_counter))

            valid_loss = self._validate(model, test_dataloader)

            # mean of epoch losses, essentially this will reflect mean batch loss for each epoch
            mean_batch_loss_training = get_mean_of_list(epoch_losses_train)

            print("Epoch:{} ------ Mean Batch Loss ({}) ------ Validation_loss: ({})".format(epoch_counter, mean_batch_loss_training, valid_loss))

            if mean_batch_loss_training < best_mean_batch_loss:
                # save the model weights
                best_mean_batch_loss = mean_batch_loss_training
                torch.save(model.state_dict(), config.SAVED_MODEL_PATH_2)
                file = os.path.join(config.MODEL_CHECKPOINT_FOLDER, 'mean_batch_loss.txt')
                with open(file, 'w') as filetowrite:
                    filetowrite.write("Epoch:{} ------ Mean Batch Loss ({}) ------ Validation_loss: ({})".format(epoch_counter, best_mean_batch_loss, valid_loss))



            if valid_loss < best_valid_loss:
                # save the model weights
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), config.SAVED_MODEL_PATH)
                file = os.path.join(config.MODEL_CHECKPOINT_FOLDER, 'validation_loss.txt')
                with open(file, 'w') as filetowrite:
                    filetowrite.write(
                        "Epoch:{} ------ Mean Batch Loss ({}) ------ Validation_loss: ({})".format(epoch_counter,
                                                                                                   mean_batch_loss_training,
                                                                                                   best_valid_loss))

            self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
            valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            #self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr(), global_step=n_iter)

    def _validate(self, model, test_dataloader):
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0

            for i_batch, sample_batched in enumerate(test_dataloader):

                xis = sample_batched[0][0].to(self.device)
                xjs = sample_batched[0][1].to(self.device)

                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss


    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = config.MODEL_CHECKPOINT_FOLDER
            state_dict = torch.load(config.SAVED_MODEL_PATH)
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model


if __name__ == '__main__':
    simclr = SimCLR()
    simclr.train()

    """

    checkpoints_folder = config.MODEL_CHECKPOINT_FOLDER

    model = ResNetSimCLR()
    model.eval()

    state_dict = torch.load(config.SAVED_MODEL_PATH, map_location=torch.device('cpu'))

    model.load_state_dict(state_dict)
    model = model.to(device)

    """

    """
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    """

    """
    # remove the projection head in the model
    new_model = nn.Sequential(*list(model.children())[:-2])
    """

    """
    for layer in new_model.children():
        print(layer)
    """

    """
    # freeze all the layers in the resnet model
    for param in new_model.parameters():
        param.requires_grad = False
        #print(param)

    new_train_transformed_dataset = ImageDataset(class_labels=training_class_labels, images=training_image_names,
                                             transform=None)
    new_train_dataloader = DataLoader(new_train_transformed_dataset, batch_size=config.BATCH_SIZE,
                                  shuffle=True, num_workers=3)

    new_test_transformed_dataset = ImageDataset(class_labels=test_class_labels, images=test_image_names,
                                            transform=None)

    new_test_dataloader = DataLoader(new_test_transformed_dataset, batch_size=config.BATCH_SIZE,
                                 shuffle=True, num_workers=3)

    """

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
    
    """
    x_train = []
    y_train = []

    for i_batch, sample_batched in enumerate(new_train_dataloader):
        img = sample_batched[0].to(device)
        outputs = new_model(img).to(device)

        x_train.extend(outputs.detach().numpy())
        y_train.extend(sample_batched[1].numpy())


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

        x_test.extend(outputs.detach().numpy())

        for j in sample_batched[1]:
            np_j = j.numpy()
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
    print("Accuracy is {}%".format(acc*100))
    """
    











