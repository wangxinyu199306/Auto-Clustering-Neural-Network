 
import tensorflow as tf
import numpy as np

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class Network(object):  
    def __init__(self): 
        #None for differeny batch size
        with tf.name_scope('input'):
            self.baseImage = tf.placeholder(tf.float32, [None,32,32,3],name='base_input')
            self.contrastImage = tf.placeholder(tf.float32, [None,32,32,3],name='contrasrt_input')
            self.labels = tf.placeholder(tf.float32,[None],name='label_input')

        with tf.variable_scope("Network") as scope:
            self.baseout = self.inference(self.baseImage)
            scope.reuse_variables()
            self.contrastout = self.inference(self.contrastImage)
        
        with tf.name_scope('loss_function'):
            self.loss = self.staticloss()

    def inference(self, images):
        images=(tf.cast(images,tf.float32)/255.-0.5)*2#归一化处理
        conv1 = self.conv_layer_3(images,32,"conv1")
        activate_1 = tf.nn.relu(conv1)
        pool1=tf.nn.max_pool(activate_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name = 'pooling1')
        conv2 = self.conv_layer_3(pool1,64,"conv2")
        activate_2 = tf.nn.relu(conv2)
        pool2=tf.nn.max_pool(activate_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name = 'pooling2')
        conv3 = self.conv_layer_3(pool2,128,"conv3")
        activate_3 = tf.nn.relu(conv3)
        flatten = tf.reshape(activate_3, [-1, 8*8*128])
        fc1 = self.fc_layer(flatten,1024,"full_c1")
        activate_5=tf.nn.relu(fc1)
        fc2 = self.fc_layer(activate_5,1024,"full_c2")
        activate_6=tf.nn.relu(fc2)
        fc3 = self.fc_layer(activate_6,2,"full_c3")
        activate_7 = tf.nn.relu(fc3)
        return activate_7

    def conv_layer_3(self, bottom , out_channels, name):
        assert len(bottom.get_shape()) == 4
        prev_channels = bottom.get_shape()[3]
        init_er = tf.truncated_normal_initializer(stddev = 0.01)
        W = tf.get_variable(name+'W', dtype = tf.float32, shape = [3,3,prev_channels,out_channels],initializer=init_er)
        variable_summaries(W)
        B = tf.get_variable(name+'B', dtype = tf.float32, initializer=tf.constant(0.01, shape = [out_channels],dtype=tf.float32))
        variable_summaries(B)
        conv_op = tf.nn.conv2d(bottom, W, strides=[1, 1, 1, 1], padding='SAME',name = name + 'matmul')
        conv_out = tf.nn.bias_add(conv_op, B,name = name + 'bias_add')
        return conv_out

    def fc_layer(self,bottom, out_count, name):
        assert len(bottom.get_shape()) == 2
        prev_channels = bottom.get_shape()[1]
        init_er = tf.truncated_normal_initializer(stddev = 0.01)
        W = tf.get_variable(name+'W', dtype = tf.float32, shape = [prev_channels,out_count],initializer=init_er)
        variable_summaries(W)
        B = tf.get_variable(name+'B', dtype = tf.float32, initializer=tf.constant(0.01, shape = [out_count],dtype=tf.float32))
        variable_summaries(B)
        mat_op = tf.matmul(bottom,W,name = name+'matmul')
        fc_out = tf.nn.bias_add(mat_op,B, name = name + 'bias_add')
        return fc_out
    
    def staticloss(self):
        margin = 6.0  #?
        labels_pos = self.labels
        labels_neg = tf.subtract(1.0, self.labels, name = "neg_label")
        educ2 = tf.reduce_sum(tf.pow(tf.subtract(self.baseout,self.contrastout),2),1)
        educ = tf.sqrt(educ2+1e-6, name = "educ_sqrt")
        C = tf.constant(margin,name="margin_assign")
        pos_loss = tf.multiply(labels_pos,educ2,name = "pos_loss_cal")
        neg_loss = tf.multiply(labels_neg,tf.pow(tf.maximum(tf.subtract(margin,educ),0),2),name = "neg_loss_cal")
        losses = tf.add(pos_loss,neg_loss,name = "loss_add")
        loss = tf.reduce_mean(losses,name="final_loss")
        return loss

    #Adam梯度下降  
    def optimer_min(self,loss,lr=0.001):
        train_optimizer = tf.train.AdamOptimizer(lr).minimize(loss)    
        return train_optimizer

    def eva_corr(self,vlad_feature, pos_feature, neg_feature):
        # vald feature =1X512, pos_feature =1X512, neg_feature =9X512
        loss_pos = tf.reduce_mean(tf.square(vlad_feature - pos_feature))
        list_neg = []
        for i in range (neg_feature.shape[0].value):
            loss_neg = tf.reduce_mean(tf.square(vlad_feature - neg_feature[i]))
            list_neg.append(loss_neg)
        return loss_pos, list_neg
        