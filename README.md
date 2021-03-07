# SimCLR_pytorch
End-to-End SimCLR model with Logistic Regression


# More Information
Parameters is followed in this paper: https://arxiv.org/pdf/2002.05709.pdf

There is no bluring of images as images for CIFAR10 is too small

Get CIFAR10 images from https://github.com/YoongiKim/CIFAR-10-images


# Run Code

Step 1: Install requirements.txt

Step 2: Ensure directories are correct in config.py

Step 3: Run "python train_simclr.py" to train the model

Step 4: Run "python run_model.py" to extract features after the model is freeze and through a Logistic Regression Classifier

