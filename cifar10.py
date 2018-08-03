import numpy as np
import cv2,random,os

 
class DataProcess(object):
    def __init__(self):
        self.train = False
        self.ROOT_DIR = "/home/lzhpc/Dataset/cifar-10-batches-py"
        self.TRAIN_DIR = os.path.join(self.ROOT_DIR, "train")
        self.TEST_DIR = os.path.join(self.ROOT_DIR, "test")
        
    def GetCifarTrainData(self):
        image_path_list = []
        label_list = []
        for class_order in range(10):
            TEMP_CLASS_DIR = os.path.join(self.TRAIN_DIR,str(class_order))
            temp_image_path_list = os.listdir(TEMP_CLASS_DIR)
            for image_path in temp_image_path_list:
                if os.path.splitext(image_path)[1] == '.jpg': 
                    image_path_list.append(os.path.join(TEMP_CLASS_DIR, image_path))
                    label_list.append(class_order)
        train_data_tuple_list = list(zip(image_path_list,label_list))
        random.shuffle(train_data_tuple_list)
        image_list,label_list = zip(*train_data_tuple_list)
        self.train_image_ndarray = np.zeros((len(image_list),32,32,3))
        self.train_label_ndarray = np.array(label_list)
        for i in range(len(image_list)):
            if((i+1)%1000 == 0):
                print("train: "+str(i+1))
            self.train_image_ndarray[i,:,:,:] = cv2.imread(image_list[i])
        
    def GetCifarTestData(self):
        image_path_list = []
        label_list = []
        for class_order in range(10):
            TEMP_CLASS_DIR = os.path.join(self.TEST_DIR,str(class_order))
            temp_image_path_list = os.listdir(TEMP_CLASS_DIR)
            for image_path in temp_image_path_list:
                if os.path.splitext(image_path)[1] == '.jpg': 
                    image_path_list.append(os.path.join(TEMP_CLASS_DIR, image_path))
                    label_list.append(class_order)
        test_data_tuple_list = list(zip(image_path_list,label_list))
        random.shuffle(test_data_tuple_list)
        image_list,label_list = zip(*test_data_tuple_list)
        self.test_image_ndarray = np.zeros((len(image_list),32,32,3))
        self.test_label_ndarray = np.array(label_list)
        for i in range(len(image_list)):
            if((i+1)%1000 == 0):
                print("test: "+str(i+1))
            self.test_image_ndarray[i,:,:,:] = cv2.imread(image_list[i])
            
    def GetTrainNextBatch(self,batch_order,total_batch):
        str_index = float(self.train_image_ndarray.shape[0])*float(batch_order)/float(total_batch)
        end_index = float(self.train_image_ndarray.shape[0])*float(batch_order + 1)/float(total_batch)
        if(int(end_index) > self.train_image_ndarray.shape[0]):
            end_index = self.train_image_ndarray.shape[0]
        return self.train_image_ndarray[int(str_index):int(end_index),:,:,:],self.train_label_ndarray[int(str_index):int(end_index)]
    
    def GetTestNextBatch(self,batch_order,total_batch):
        str_index = float(self.test_image_ndarray.shape[0])*float(batch_order)/float(total_batch)
        end_index = float(self.test_image_ndarray.shape[0])*float(batch_order + 1)/float(total_batch)
        if(int(end_index) > self.test_image_ndarray.shape[0]):
            end_index = self.test_image_ndarray.shape[0]
        return self.test_image_ndarray[int(str_index):int(end_index),:,:,:],self.test_label_ndarray[int(str_index):int(end_index)]

