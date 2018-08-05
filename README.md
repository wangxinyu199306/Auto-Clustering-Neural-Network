# Auto-Clustering-Neural-Network
CNN for weak supervised clustering

# Data Structure
In this case, cifar10 was used for training and testing, the structure of dataset is organized as following:

________________folder:train -> folder: 0,1,2,3,4,5,6,7,8,9
__folder:data->               
________________folder:test  -> folder: 0,1,2,3,4,5,6,7,8,9
              
each folder i(i=0,1,2,...,9) contains all the images with label i, 5000 images for each folder in training, 1000 for testing.

# Network
A simple net with 3 conv layer & 3 fc layer is used in this demo, which I believe it's suit enough for cifar10.
![image](https://github.com/wangxinyu199306/Auto-Clustering-Neural-Network/blob/master/structure.png)       


# Run it
change the dataset path in cifar10.py(ignore the config.py), and run with:
python3 cifar10_strtucture
following image will be shown after the program finished:
![image](https://github.com/wangxinyu199306/Auto-Clustering-Neural-Network/blob/master/predict.png)
A better performance can be approached by substitute 6.0 with a larger margin value in featureNet.py/

# Loss during training
![image](https://github.com/wangxinyu199306/Auto-Clustering-Neural-Network/blob/master/loss.png)
