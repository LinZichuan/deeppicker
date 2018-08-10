import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tcl
import sys
#sys.path.append("./YellowFin/tuner_utils")
#from yellowfin import YFOptimizer
    
def batch_normalization_layer(input_layer, dimension, phase):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, 0.001)
    '''
    bn_layer = tcl.batch_norm(input_layer, center=True, scale=True, is_training=phase, scope='bn')

    return bn_layer

def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''
    
    ## TODO: to allow different weight decay to fully connected layer and conv layer
    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0002)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0002)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        #batch norm
        #axis = list(range(len(x.get_shape()) - 1))
        #mean, variance = tf.nn.moments(x, axis)
        #x = tf.nn.batch_normalization(x, mean, variance, 0, 1, 0.001 )
        f1 = 0.5 * (1+leak)
        f2 = 0.5 * (1-leak)
        return f1 * x + f2 * abs(x)

def conv_bn_relu_layer(input_layer, filter_shape, stride, phase):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''
    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel, phase=phase)

    output = tf.nn.relu(bn_layer)
    return output

def bn_relu_conv_layer(input_layer, filter_shape, stride, phase):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''
    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel, phase=phase)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer

def residual_block(input_layer, output_channel, first_block=False, phase=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            #filter = tf.get_variable(name='conv', shape=[3, 3, input_channel, output_channel],
            #                initializer = tf.truncated_normal_initializer(stddev=0.05, seed = 1234))
            #conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride, phase)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1, phase)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output

class DeepModel():
    def __init__(self, particle_size, model_input_size, num_class, dropout_rate=0.5, finetune=False):
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
        '''
        self.w[1] = self.__variable('w1', shape=[ks[0], ks[0], 1, fm[0]], stddev=0.05, wd=0.0)
        self.b[1] = tf.get_variable('b1', [fm[0]], initializer=tf.constant_initializer(0.0))
        self.w[2] = self.__variable('w2', shape=[ks[1], ks[1], fm[0], fm[1]], stddev=0.05, wd=0.0)
        self.b[2] = tf.get_variable('b2', [fm[1]], initializer=tf.constant_initializer(0.0))
        self.w[3] = self.__variable('w3', shape=[ks[2], ks[2], fm[1], fm[2]], stddev=0.05, wd=0.0)
        self.b[3] = tf.get_variable('b3', [fm[2]], initializer=tf.constant_initializer(0.0))
        self.w[4] = self.__variable('w4', shape=[ks[3], ks[3], fm[2], fm[3]], stddev=0.05, wd=0.0)
        self.b[4] = tf.get_variable('b4', [fm[3]], initializer=tf.constant_initializer(0.0))
        '''

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


        '''
        conv1 = tf.nn.conv2d(self.data, self.w1, strides=[1,1,1,1], padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, self.b1))
        pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        print pool1.get_shape()

        conv2 = tf.nn.conv2d(pool1, self.w2, strides=[1,1,1,1], padding='VALID')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, self.b2))
        pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        print pool2.get_shape()

        conv3 = tf.nn.conv2d(pool2, self.w3, strides=[1,1,1,1], padding='VALID')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, self.b3))
        pool3 = tf.nn.max_pool(relu3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        print pool3.get_shape()

        conv4 = tf.nn.conv2d(pool3, self.w4, strides=[1,1,1,1], padding='VALID')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, self.b4))
        pool4 = tf.nn.max_pool(relu4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        print pool4.get_shape()
        '''

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

        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        #        fc2, self.label, name='cross_entropy_all')
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                fc2, self.label, name='cross_entropy_all')
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
