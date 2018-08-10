import tensorflow as tf
import numpy as np
from optparse import OptionParser
from deepModel import DeepModel
from dataLoader import DataLoader
import time, os
import random
from tqdm import tqdm


def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    newa = []
    newb = []
    for i in range(len(a)):
        newa.append(a[p[i]])
        newb.append(b[p[i]])
    #return a[p], b[p]
    return newa, newb

def error_rate(prediction, label):
    """Return the error rate based on dense predictions and sparse labels."""
    #label = [int(l[1]) for l in label]
    return 100.0 - (100.0 * np.sum(np.argmax(prediction, 1) == label) / prediction.shape[0])

def train():
    parser = OptionParser()
    parser.add_option("--train_inputDir", dest="train_inputDir", 
            help="Input directory", metavar="DIRECTORY")
    parser.add_option("--train_inputFile", dest="train_inputFile", 
            help="Input file", metavar="FILE")
    parser.add_option("--train_type", dest="train_type", 
            help="Training type, 1|2|3|4.", metavar="VALUE", default=2)
    parser.add_option("--particle_number", dest="train_number", 
            help="Number of positive samples to train.", metavar="VALUE", default=-1)
    parser.add_option("--mrc_number", dest="mrc_number", 
            help="Number of mrc files to be trained.", metavar="VALUE", default=-1)
    parser.add_option("--coordinate_symbol", dest="coordinate_symbol", 
            help="The symbol of the coordinate file, like '_manualPick'", metavar="STRING")
    parser.add_option("--particle_size", dest="particle_size", 
            help="the size of the particle.", metavar="VALUE", default=-1)
    parser.add_option("--validation_ratio", dest="validation_ratio", 
            help="the ratio.", metavar="VALUE", default=0.1)
    parser.add_option("--model_retrain", action="store_true", dest="model_retrain", 
            help="train the model using the pre-trained model as parameters initialization .", default=False)
    parser.add_option("--model_load_file", dest="model_load_file", 
            help="pre-trained model", metavar="FILE")
    parser.add_option("--model_save_dir", dest="model_save_dir", 
            help="save the model to this directory", metavar="DIRECTORY", default="../trained_model")
    parser.add_option("--model_save_file", dest="model_save_file", 
            help="save the model to file", metavar="FILE")
    parser.add_option("--pos_list", dest="pos_list",
            help="", metavar="VALUE", default="")
    parser.add_option("--neg_list", dest="neg_list",
            help="", metavar="VALUE", default="")

    parser.add_option("--mixup", dest="mixup",
            help="", metavar="VALUE", default="0")
    (opt, args) = parser.parse_args()
    
    model_input_size = [128, 64, 64, 1]
    num_class = 2
    batch_size = model_input_size[0]
    # define input parameters
    train_type = int(opt.train_type)
    train_inputDir = opt.train_inputDir
    train_inputFile = opt.train_inputFile
    protein_number = len(train_inputFile.split(';'))
    train_number = float(opt.train_number) 
    mrc_number = int(opt.mrc_number)
    dropout_rate = 0.5
    coordinate_symbol = opt.coordinate_symbol
    debug_dir = '../train_output'   # output dir
    particle_size = int(opt.particle_size)
    validation_ratio = float(opt.validation_ratio)
    # define the save model
    model_retrain = opt.model_retrain
    model_load_file = opt.model_load_file
    model_save_dir = opt.model_save_dir
    model_save_file = os.path.join(model_save_dir, opt.model_save_file)
    pos_list = opt.pos_list
    neg_list = opt.neg_list
    mixup = int(opt.mixup)
    print ("MIXUP=======", mixup)
    if not os.access(model_save_dir, os.F_OK):
        os.mkdir(model_save_dir)
    if not os.access(debug_dir, os.F_OK):
        os.mkdir(debug_dir)
    dataLoader = DataLoader()

    train_number = int(train_number)
    if train_type == 1:
        # load train data from mrc file dir
        train_data, train_label, eval_data, eval_label = dataLoader.load_trainData_From_mrcFileDir(
                train_inputDir, particle_size, model_input_size, validation_ratio, 
                coordinate_symbol, mrc_number, train_number)
    elif train_type == 2:
        # load train data from numpy data struct
        train_data, train_label, eval_data, eval_label = dataLoader.load_trainData_From_ExtractedDataFile(
                train_inputDir, train_inputFile, model_input_size, validation_ratio, train_number)
    elif train_type == 3:
        # load train data from prepicked results
        train_data, train_label, eval_data, eval_label = dataLoader.load_trainData_From_PrePickedResults(
                train_inputDir, train_inputFile, particle_size, model_input_size, validation_ratio, train_number)
    elif train_type == 4:
        # load train data from relion .star file 
        train_data, train_label, eval_data, eval_label = dataLoader.load_trainData_From_RelionStarFile(
                train_inputFile, particle_size, model_input_size, validation_ratio, train_number)
    elif train_type == 5:
        # load train data from class2d .star file
        train_data, train_label, eval_data, eval_label = dataLoader.load_trainData_From_Class2dStarFile(
                train_inputDir, train_inputFile, model_input_size, validation_ratio, train_number)
    elif train_type == 6:
        left = 0
        right = 50
        get_partition = lambda x, y: (x + y) / 2
        '''
        # load train data from auto_filter_class .star file
        train_data, train_label, eval_data, eval_label = dataLoader.load_trainData_From_AutoClass2dStarFile(
                train_inputDir, train_inputFile, model_input_size, validation_ratio, train_number)
        '''
    else:
        print("ERROR: invalid value of train_type:", train_type)   
    try:
        train_type==6 or train_data
    except NameError:
        print("Error: in function load.loadInputTrainData.")
        return None
    else:
        print("Load training data successfully!")

    idx = 0
    good_enough = False
    while True and not good_enough:
        best_eval_error_rate = 100
        all_error = []
        finetune = False if train_type == 6 else False
        dropout_rate = 0.5 if train_type == 6 else dropout_rate
        deepModel = DeepModel(particle_size, model_input_size, num_class, dropout_rate=dropout_rate, finetune=finetune)
        if train_type == 6:
            deepModel.learning_rate = deepModel.learning_rate / 10.0
            deepModel.decay_steps *= 2
            if good_enough:
                partition = partition + 1
            else:
                partition = get_partition(left, right)
            print "PARTITOIN --->>>", partition
            partition = 9
            good_enough = True  #Set this=True to run while for just once!!!
            #train_data, train_label, eval_data, eval_label = dataLoader.load_trainData_From_AutoClass2dStarFile(train_inputDir, train_inputFile, model_input_size, validation_ratio, train_number, partition)
            train_data, train_label, eval_data, eval_label = dataLoader.load_trainData_From_AutoClass2dStarFile(train_inputDir, train_inputFile, model_input_size, validation_ratio, train_number, partition, pos_list, neg_list)
        train_data, train_label = shuffle_in_unison_inplace(train_data, train_label)
        print ("label_shape = ", np.array(train_label).shape)
        '''
        mix_data, mix_label = [], []
        if mixup:
            mixnum = len(train_data)
            #for cnt in tqdm(range(mixnum)):
            for cnt in range(mixnum):
            #for cnt in range(mixnum):
                L = np.random.beta(0.2, 0.2)
                i1, i2 = np.random.randint(mixnum, size=2)
                if train_data[i1].shape[1] == train_data[i2].shape[1]:
                    new_data = (1-L) * train_data[i1] + L * train_data[i2]
                    new_label = (1-L) * train_label[i1][1] + L * train_label[i2][1]
                    mix_data.append(new_data)
                    mix_label.append([1.0-new_label, new_label])
        train_data = train_data + mix_data
        train_label = train_label + mix_label
        '''
        print ("label_shape = ", np.array(train_label).shape)
        #eval_data, eval_label = shuffle_in_unison_inplace(eval_data, eval_label)
        bs2train = {}
        bs2eval = {}
        for idx,t in enumerate(train_data):
            if t.shape[1] not in bs2train.keys():
                bs2train[t.shape[1]] = [idx]
            else:
                bs2train[t.shape[1]].append(idx)
        for idx,t in enumerate(eval_data):
            if t.shape[1] not in bs2eval.keys():
                bs2eval[t.shape[1]] = [idx]
            else:
                bs2eval[t.shape[1]].append(idx)
        train_size = len(train_data)
        eval_size = len(eval_data)
        print ("train size=%d, eval_size=%d" % (train_size, eval_size))
        print ("batch_size=%d" % batch_size)
        print ("dropout=%.2f" % dropout_rate)
        if train_size < 1000:
            print ("NOTE: no enough training data!\n<Failed>! ")
            exit()
        '''
        if eval_size < model_input_size[0]: #TODO
            tile_size = model_input_size[0] // eval_size + 1
            eval_data = np.array(eval_data)
            eval_data = np.tile(eval_data, [tile_size,1,1,1])
            print ("tiled eval_data !!!!", tile_size)
        '''
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=30)
        
        start_time = time.time()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.26)
        train_error = []
        valid_error = []
        eval_time = 0
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)) as sess:
            tf.initialize_all_variables().run()
            if model_load_file:
                print model_load_file
                saver.restore(sess, model_load_file)
            max_epochs = 200
            best_eval_error_rate = 100
            toleration_patience = 10
            toleration_patience_flag  = 0
            eval_frequency = train_size // batch_size
            print ("total_step=%d" % (int(max_epochs * train_size) // batch_size))
            #fout = open('trainingcurve%d_%s_test2_block1_lr0.1.txt'%(protein_number, deepModel.arch), 'w')
            #fout = open('trainingcurve%d_%s_lr0.1.txt'%(protein_number, deepModel.arch), 'w')
            #fout = open('trainingcurve%d_resnet.txt'%protein_number, 'w')
            idx += 1
            batch_type = bs2train.keys()
            batch_type_number = len(batch_type)
            po = {}
            for k in range(batch_type_number):
                po[k] = 0
            batch_type_idx = 0
            train_error_list = []
            print ("===================================================================")
            #for step in xrange(int(max_epochs * train_size) // batch_size):
            eval_prediction = deepModel.evaluation(eval_data, sess, label=eval_label)
            eval_error_rate = error_rate(eval_prediction, eval_label)
            eval_before_retrain = eval_error_rate
            print('valid error before training: %.6f%%' % eval_error_rate)
            print ("===================================================================")
            for epoch in range(int(max_epochs)):
                start_time = time.time()
                #for s in tqdm(range(eval_frequency)):
                for s in range(eval_frequency):
                    step = epoch * eval_frequency + s
                    # get the batch training data
                    offset =  (step * batch_size) % (train_size - batch_size)
                    batch_type_idx = (batch_type_idx + 1) % batch_type_number
                    batch = batch_type[batch_type_idx]
                    if po[batch_type_idx] + batch_size >= len(bs2train[batch]):
                        po[batch_type_idx] = 0
                    p = po[batch_type_idx]
                    idxs = bs2train[batch][p:(p+batch_size)]
                    batch_data = []
                    batch_label = []
                    for ix in idxs:
                        batch_data.append(train_data[ix])
                        batch_label.append(train_label[ix])
                    po[batch_type_idx] = po[batch_type_idx] + batch_size
                    #batch_data = train_data[offset:(offset+batch_size)]
                    #batch_label = train_label[offset:(offset+batch_size)]
                    '''
                    batch_data_shape = batch_data[0].shape
                    con = False
                    for bb in batch_data:
                        if bb.shape != batch_data_shape:
                            con = True
                            break
                    if con:
                        continue
                    '''
                    # online augmentation
                    #batch_data = DataLoader.preprocess_particle_online(batch_data)
                    loss_value, lr, train_prediction = deepModel.train_batch(batch_data, batch_label, sess)
                    train_error_list.append(error_rate(train_prediction, batch_label))
                    
                    # do the computation
                    #if step % eval_frequency == 0:
                    #if step % 50 == 0:
                #TODO:display after each epoch
                stop_time = time.time() - start_time
                eval_prediction = deepModel.evaluation(eval_data, sess, label=eval_label)
                eval_error_rate = error_rate(eval_prediction, eval_label)
                #best_eval_error_rate = min(best_eval_error_rate, eval_error_rate)
                #print('>> epoch: %.2f , %.2f ms' % (step * batch_size /train_size, 1000 * stop_time / eval_frequency)) 
                train_error_mean = np.mean(train_error_list)
                print('>> epoch: %d, train loss: %.2f, lr: %.6f, toleration:%d, train error: %.2f%%, valid error: %.2f%%' % (epoch, loss_value, lr, toleration_patience, train_error_mean, eval_error_rate)) 
                #print >>fout, step, train_error_mean, eval_error_rate
                train_error.append(train_error_mean)
                valid_error.append(eval_error_rate)
                eval_time += 1
                train_error_list = []
                all_error.append(eval_error_rate)

                if eval_error_rate < best_eval_error_rate:
                    best_eval_error_rate = eval_error_rate
                    toleration_patience = 10
                    saver.save(sess, model_save_file)
                else:
                    if epoch > 50:
                        toleration_patience = toleration_patience - 1

                if toleration_patience == 0:
                    break
        good_enough = True
        '''
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.title('Training curve')
        plt.ylabel('Error(%)')
        plt.xlabel('Epoch')
        axes = plt.gca()
        axes.set_ylim([0, 60])
        plt.plot(range(eval_time), train_error, label='training')
        plt.plot(range(eval_time), valid_error, label='validation')
        plt.legend(loc='upper right')
        plt.show()
        #plt.savefig('pickercurve.png')
        '''
        print ("Accuracy: before retrain: %.2f%%, after retrain: %.2f%%" % (100.0-eval_before_retrain, 100.0-best_eval_error_rate))
        print ("Retrain <Successful>!")

if __name__ == '__main__':
    train()
