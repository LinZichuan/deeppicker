import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tcl
import sys
from utils import *
#sys.path.append("./YellowFin/tuner_utils")
#from yellowfin import YFOptimizer

class DeepModel():
    def __init__(self, particle_size, model_input_size, num_class, dropout_rate=0.5, finetune=False, maml=False):
        self.maml = maml
        if self.maml:
            self.update_lr = 0.01
            self.meta_lr = 0.01
        self.weights = {}
        self.particle_size = particle_size
        self.batch_size = model_input_size[0]
        self.num_col = model_input_size[1]
        self.num_row = model_input_size[2]
        self.num_channel = model_input_size[3]
        self.num_class = num_class
        self.global_step = tf.Variable(0, trainable=False)
        #self.arch = 'default'
        self.arch = 'improved'
        #self.arch = 'deeper'
        #self.arch = 'resnet'
        self.dropout_rate = dropout_rate
        self.finetune = finetune
        if self.arch == 'default':
            self.learning_rate = 0.01
            self.learning_rate_decay = 0.95
            self.decay_steps = 400
            self.momentum = 0.9
        else:
            self.learning_rate = 0.01
            self.learning_rate_decay = 0.95
            self.decay_steps = 800
            self.momentum = 0.9
        self.verbose = False
        self.build_graph()

    def __variable(self, name, shape, stddev, wd, trainable=True):
        var = tf.get_variable(name, shape,
                        initializer = tf.truncated_normal_initializer(stddev=stddev, seed = 1234), trainable=trainable)
        if wd is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def build_graph(self):
        if self.arch == 'default':
            ks = [9, 5, 3, 2] # 9,5,3,2
        elif self.arch == 'improved':
            ks = [5, 5, 3, 2] # 9,5,3,2
            self.ks = ks
        elif self.arch == 'deeper':
            ks = [3,3,3,3,3,3]
        elif self.arch == 'resnet':
            num_block = 1
        #ks = [3, 3, 3, 3] # 9,5,3,2
        if self.arch == 'deeper':
            fm = [1, 8, 16, 32, 32, 32, 64]
        elif self.arch == 'resnet':
            fm = [1, 8, 16, 32]
        else:
            fm = [1, 8, 16, 32, 64]
        self.w = {}
        self.b = {}
        self.data = tf.placeholder('float32', [None, self.num_col, self.num_row, self.num_channel], name='data')
        #self.label = tf.placeholder('float32', shape=(self.batch_size,2))
        self.label = tf.placeholder('int64', shape=(self.batch_size,))
        self.is_train = tf.placeholder('int64', None, name='is_train')
        if self.is_train == 1:
            phase = True
        else:
            phase = False
        data = self.data

        ####### CONV LAYER
        if self.arch == 'resnet':
            with tf.variable_scope('conv0'):
                data = conv_bn_relu_layer(data, [3, 3, 1, fm[1]], 1, phase)
            # total layers = 1 + 2n + 2n + 2n + 1 = 6n + 2
            for i in range(len(fm)-1):
                for j in range(num_block):
                    with tf.variable_scope('conv%d_%d'%(i,j)):
                        data = residual_block(data, fm[i+1], first_block = i==0 and j==0, phase=phase)
            bn_layer = batch_normalization_layer(data, fm[-1], phase=phase)
            relu_layer = tf.nn.relu(bn_layer)
            if self.verbose:
                print ("begore global pool:", relu_layer.get_shape())
            global_pool = tf.reduce_mean(relu_layer, [1, 2])
            data = global_pool
            if self.verbose:
                print ("after global pool:", data.get_shape())
        else:
            if not self.finetune:
                for i in range(len(ks)):
                    self.w[i] = self.__variable('w%d'%i, shape=[ks[i], ks[i], fm[i], fm[i+1]], stddev=0.05, wd=0.0)
                    self.b[i] = tf.get_variable('b%d'%i, [fm[i+1]], initializer=tf.constant_initializer(0.0))
                    if self.maml:
                        self.weights['w'+str(i)] = self.w[i]
                        self.weights['b'+str(i)] = self.b[i]
            else:
                for i in range(len(ks)): #NOTE:if finetune, only train the highest conv layer
                    trainable = True if i == len(ks)-1 else False
                    self.w[i] = self.__variable('w%d'%i, shape=[ks[i], ks[i], fm[i], fm[i+1]], stddev=0.05, wd=0.0, trainable=trainable)
                    self.b[i] = tf.get_variable('b%d'%i, [fm[i+1]], initializer=tf.constant_initializer(0.0), trainable=trainable)
            for i in range(len(ks)):
                if self.arch == 'deeper':
                    stride = 1 if i%2==0 else 2
                    conv = tf.nn.conv2d(data, self.w[i], strides=[1,stride,stride,1], padding='VALID')
                else:
                    conv = tf.nn.conv2d(data, self.w[i], strides=[1,1,1,1], padding='VALID')

                if self.arch == 'default':
                    relu = tf.nn.relu(tf.nn.bias_add(conv, self.b[i]))
                else:
                    relu = lrelu(tf.nn.bias_add(conv, self.b[i]))

                if self.arch == 'deeper':
                    data = relu
                else:
                    pool = tf.nn.max_pool(relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
                    data = pool
                if self.verbose:
                    print data.get_shape()


        ####### FC LAYER

        #shape = data.get_shape().as_list()
        #dim = reduce(lambda x,y: x*y, shape)
        if self.arch == 'deeper':
            dim = fm[-1] * 5 * 5
            self.w_fc1 = self.__variable('wfc1', shape=[dim, self.num_class], stddev=0.05, wd=0.0005)
            self.b_fc1 = tf.get_variable('bfc1', [self.num_class], initializer=tf.constant_initializer(0.0))
        elif self.arch == 'resnet':
            dim = fm[-1] * 1 * 1 # 64x16x16 -> global pool -> 64x1x1
            self.w_fc1 = self.__variable('wfc1', shape=[dim, self.num_class], stddev=0.05, wd=0.0005)
            self.b_fc1 = tf.get_variable('bfc1', [self.num_class], initializer=tf.constant_initializer(0.0))
        else:
            dim = fm[-1] * 2 * 2
            self.w_fc1 = self.__variable('wfc1', shape=[dim, 64], stddev=0.05, wd=0.0005)
            self.b_fc1 = tf.get_variable('bfc1', [64], initializer=tf.constant_initializer(0.0))
            self.w_fc2 = self.__variable('wfc2', shape=[64, self.num_class], stddev=0.05, wd=0.0005)
            self.b_fc2 = tf.get_variable('bfc2', [self.num_class], initializer=tf.constant_initializer(0.0))
            if self.maml:
                self.weights['w_fc1'] = self.w_fc1
                self.weights['b_fc1'] = self.b_fc1
                self.weights['w_fc2'] = self.w_fc2
                self.weights['b_fc2'] = self.b_fc2

        #hid = tf.reshape(pool4, [self.batch_size, -1])
        hid = tf.reshape(data, [self.batch_size, -1])
        if self.is_train == 1:
            hid = tf.nn.dropout(hid, self.dropout_rate)

        #fc1 = tf.nn.relu(tf.matmul(hid, self.w_fc1) + self.b_fc1)
        if self.arch == 'default':
            fc1 = tf.nn.relu(tf.matmul(hid, self.w_fc1) + self.b_fc1)
            fc2 = tf.matmul(fc1, self.w_fc2) + self.b_fc2
        elif self.arch == 'improved':
            fc1 = lrelu(tf.matmul(hid, self.w_fc1) + self.b_fc1)
            fc2 = tf.matmul(fc1, self.w_fc2) + self.b_fc2
        elif self.arch == 'deeper':
            fc2 = tf.matmul(hid, self.w_fc1) + self.b_fc1
        elif self.arch == 'resnet':
            fc2 = tf.matmul(hid, self.w_fc1) + self.b_fc1

        ####### SOFTMAX LAYER
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(fc2, self.label, name='cross_entropy_all')
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(fc2, self.label, name='cross_entropy_all')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        self.loss = tf.add_n(tf.get_collection('losses'), name='loss')
        self.learning_rate_op = tf.maximum(tf.train.exponential_decay(
                self.learning_rate, self.global_step, self.decay_steps, 
                self.learning_rate_decay, staircase=True), 0.0001)
        self.optim = tf.train.MomentumOptimizer(self.learning_rate_op, self.momentum).minimize(self.loss, global_step=self.global_step)
        #self.optim = tf.train.MomentumOptimizer(0.01, self.momentum).minimize(self.loss, global_step=self.global_step)
        #self.optim = YFOptimizer().minimize(self.loss, global_step=self.global_step)
        #self.optim = tf.train.AdamOptimizer(self.learning_rate_op).minimize(self.loss, global_step=self.global_step)
        self.prediction_op = tf.nn.softmax(fc2)

        #tf.initialize_all_variables().run()
        self.bdata = tf.placeholder('float32', [None, None, None, self.num_channel], name='bdata')
        resize_size = (self.num_col, self.num_row)
        #self.resize_op = tf.image.resize_bilinear(self.bdata, resize_size)

        img = tf.image.resize_bilinear(self.bdata, resize_size)
        axis = list(range(len(img.get_shape()) - 1))
        mean, variance = tf.nn.moments(img, axis)
        self.resize_op = tf.nn.batch_normalization(img, mean, variance, 0, 1, 0)

        params_number = self.get_trainable_variable_number()
        #if self.verbose:
        print "parameter number ======================= ", params_number

    def get_trainable_variable_number(self):
        cnt = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            if self.verbose:
                print(shape)
            var_params = 1
            for dim in shape:
                var_params *= dim.value
            if self.verbose:
                print (var_params)
            cnt += var_params
        return cnt

    def train_batch(self, data, label, sess):
        data = sess.run(self.resize_op, feed_dict={self.bdata: data})
        _, loss_value, learning_rate, prediction = sess.run(
                [self.optim, self.loss, self.learning_rate_op, self.prediction_op], 
                feed_dict={self.data: data, self.label: label, self.is_train: 1}
        )
        return loss_value, learning_rate, prediction

    def metatrain_batch(self, data, label, sess):
        #data: [batch_size, num_samples, dim_input]
        data = sess.run(self.resize_op, feed_dict={self.bdata: data})
        _, loss_value, learning_rate, prediction = sess.run(
                [self.optim, self.loss, self.learning_rate_op, self.prediction_op], 
                feed_dict={self.data: data, self.label: label, self.is_train: 1}
        )

        #task_outputa = self.forward(data, self.weights, reuse=reuse)
        #task_lossa = self.loss_func(task_outputa, label)
        #Use Weights To Get Loss And Gradients 
        task_outputa = self.loss
        grads = tf.gradients(task_outputa, list(self.weights.values()))

        if self.stop_grad:
            grads = [tf.stop_gradient(grad) for grad in grads]
        gradients = dict(zip(self.weights.keys(), grads))
        fast_weights = dict(zip(self.weights.keys(), [self.weights[key] - self.update_lr*gradients[key] for key in self.weights.keys()]))
        #output = self.forward(inputb, fast_weights, reuse=True)
        #task_outputbs.append(output)
        #task_lossesb.append(self.loss_func(output, labelb))

        return loss_value, learning_rate, prediction

    def forward(inp, weights, reuse=False):
        for i in range(len(self.ks)):
            conv = tf.nn.conv2d(data, weights['w'+str(i)], strides=[1,1,1,1], padding='VALID')
            relu = lrelu(tf.nn.bias_add(conv, weights['b'+str(i)]))
            data = tf.nn.max_pool(relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        hid = tf.reshape(data, [self.batch_size, -1])
        if self.is_train == 1:
            hid = tf.nn.dropout(hid, self.dropout_rate)
        fc1 = lrelu(tf.matmul(hid, weights['w_fc1']) + weights['b_fc1'])
        fc2 = tf.matmul(fc1, weights['w_fc2']) + weights['b_fc2']

    def evaluation(self, data, sess, label=None):
        size = len(data)
        predictions = np.ndarray(shape=(size, self.num_class), dtype=np.float32)
        #resize_size = (self.num_col, self.num_row)
        #data = tf.image.resize_bilinear(data, resize_size).eval()
        #data = sess.run(self.resize_op, feed_dict={self.bdata: data})
        for begin in xrange(0, size, self.batch_size):
            end = begin + self.batch_size
            if end <= size:
                batch_data = data[begin:end]
                if batch_data[0].shape != batch_data[-1].shape:
                    predictions[begin:end, :] = label[begin:end]
                    continue
                #batch_data = tf.image.resize_bilinear(batch_data, resize_size).eval()
                #mean = batch_data.mean()
                #std = batch_data.std()
                #batch_data = (batch_data - mean) / std
                batch_data = sess.run(self.resize_op, feed_dict={self.bdata: batch_data})
                predictions[begin:end, :] = sess.run(
                    self.prediction_op,
                    feed_dict={self.data: batch_data})
            else:
                batch_data = data[-self.batch_size:]
                if batch_data[0].shape != batch_data[-1].shape:
                    predictions[begin:end, :] = label[begin:end]
                    continue
                #batch_data = tf.image.resize_bilinear(batch_data, resize_size).eval()
                #mean = batch_data.mean()
                #std = batch_data.std()
                #batch_data = (batch_data - mean) / std
                batch_data = sess.run(self.resize_op, feed_dict={self.bdata: batch_data})
                batch_predictions = sess.run(
                    self.prediction_op,
                    feed_dict={self.data: batch_data})
                predictions[begin:, :] = batch_predictions[(begin-(size-self.batch_size)):, :]
        return predictions
