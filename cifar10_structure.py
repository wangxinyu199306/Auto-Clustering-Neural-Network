
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
import cv2,random,os
from cifar10 import DataProcess
from featureNet import Network
from visual import visual_result

dp = DataProcess()
dp.GetCifarTrainData()
dp.GetCifarTestData()


lr = tf.placeholder(tf.float32,name = 'learning_rate')
network  = Network()
loss = network.staticloss()
tf.summary.scalar('loss',loss)
train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
writer = tf.summary.FileWriter('logs/',tf.get_default_graph())
saver = tf.train.Saver()
learning_rate = 0.01


merged = tf.summary.merge_all()
#batch_size = 100
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    
    for epoch in range(80):
        learning_rate*=0.98
        for batch in range(500):
            image_x1,label_x1 = dp.GetTrainNextBatch(batch,500)
            which_batch = random.randint(0, 499)
            while which_batch == batch :
                which_batch = random.randint(0, 499)
            image_x2,label_x2 = dp.GetTrainNextBatch(which_batch,500)
            # print(str(batch) + " "+ str(which_batch) + " ", end='')
            # print(label_x1.shape, end='')
            # print(label_x2.shape)
            label_contrast = (label_x1 == label_x2).astype('float')
            summary_all,output,loss_temp,_=sess.run([merged,network.baseout,network.loss,train_step], feed_dict = {network.baseImage:image_x1,\
             network.contrastImage:image_x2,network.labels:label_contrast, lr:learning_rate})
            if (batch+1) % 50 == 0:
                print ('batch %d: loss %.3f' % (batch+1, loss_temp))
                #print(output[0,:])
        writer.add_summary(summary_all,epoch)
        average_loss = 0.0
        for test_batch in range(10):
            image_test,label_test = dp.GetTestNextBatch(2*test_batch,20)
            image_test_2,label_test_2 = dp.GetTestNextBatch(2*test_batch+1,20)
            test_contrast = (label_test == label_test_2).astype('float')
            temp_loss = sess.run([network.loss], feed_dict = {network.baseImage:image_test,network.contrastImage:image_test_2,network.labels:test_contrast})
            average_loss += temp_loss[0]
        print('epoch %d: loss %.4f' % (epoch,average_loss/10))
        saver.save(sess,'./model')


    list_image = []
    list_pos = []
    for test_batch in range(1000):
        image_test,label_test = dp.GetTestNextBatch(test_batch,10000)
        temp_pos = sess.run([network.baseout], feed_dict = {network.baseImage:image_test})
        list_image.append(image_test[0])
        list_pos.append(temp_pos[0])
    print(len(list_pos))

    image_ndarray = np.array(list_image)#1000,32,32,3 float64
    pos_ndarray = np.array(list_pos) #1000,1,2 float32
    print(pos_ndarray.shape)
    visual_result(image_ndarray, pos_ndarray)
    #visual(np.array(list_image),np.array(list_pos))
