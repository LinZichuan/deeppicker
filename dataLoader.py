import os
import sys
import struct
from PIL import Image
#from pylab import *
import numpy as np
import re
import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt
import scipy.misc
import scipy.ndimage
import tensorflow as tf
import random
from operator import itemgetter, attrgetter
from matplotlib import pyplot as plt
from numpy import *

import display 
from starReader import starRead
import skimage
from skimage.transform import resize

class DataLoader(object):
    
    #def __init__(self):
  
    @staticmethod 
    def bin_2d(body_2d, bin_size):
        """Do the bin process to the 2D array.

        This function can make bin the image based on the bin_size.
        bin_size is a int value. if it was set to 2, then the 4 points in a small patch 2x2 of the body_2d
               are summed to one value. It likes an average pooling operation.  

        Args:
            body_2d: numpy.array, it is a 2d array, the dim is 2.
            bin_size: int value. 

        Returns:
            return pool_result
            pool_result: numpy.array, the shape of it is (body_2d.shape[0]/bin_size, body_2d.shape[1]/bin_size)
        
        """
        """
        # using the tensorflow pooling operation to do the bin preprocess
        # memory cost, out of memory
        col = body_2d.shape[0]
        row = body_2d.shape[1]
        body_2d = body_2d.reshape(1, col, row, 1)
        body_node = tf.constant(body_2d)
        body_pool = tf.nn.avg_pool(body_node, ksize=[1, bin_size, bin_size, 1], strides=[1, bin_size, bin_size, 1], padding='VALID')
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            pool_result = sess.run(body_pool)
            pool_result = pool_result.reshape((pool_result.shape[1], pool_result.shape[2]))
        return pool_result
        """
        # based on the numpy operation to do the bin process
        col = body_2d.shape[0]
        row = body_2d.shape[1]
        scale_col = col//bin_size
        scale_row = row//bin_size
        patch = np.copy(body_2d[0:scale_col*bin_size, 0:scale_row*bin_size])
        patch_view = patch.reshape(scale_col, bin_size, scale_row, bin_size)
        #body_2d_bin = patch_view.mean(axis=3).mean(axis=1)
        body_2d_bin = patch_view.sum(axis=3).sum(axis=1)
        return body_2d_bin
 
    @staticmethod
    def preprocess_micrograph(micrograph):
        """Do preprocess to the micrograph after the micrograph data is loaded into a numpy.array.

        Define this function to make sure that the same process is done to the micrograph 
            during the training process and picking process.
        
        Args:
            micrograph: numpy.array, the shape is (micrograph_col, micrograph_row)

        Returns:
            return micrograph
            micrograph: numpy.array
        """
        mrc_col = micrograph.shape[0]
        mrc_row = micrograph.shape[1]
        k = 100
        # lowpass
        #print micrograph.min(), micrograph.max()
        #micrograph = scipy.ndimage.filters.gaussian_filter(micrograph, 0.1) 
        # do the bin process 
        pooling_size = 4
        #print micrograph[0:4, 0:4].sum()
        micrograph = DataLoader.bin_2d(micrograph, pooling_size)
        #print micrograph[0:1, 0:1]
        mean = micrograph.mean()
        std = micrograph.std()
        truncate = 4.0
        min_ = mean - truncate * std
        max_ = mean + truncate * std
        #print (min_, max_)
        sortmrc = np.sort(micrograph.reshape(-1))
        min_ = sortmrc[k]
        max_ = sortmrc[-k]
        #print (min_, max_)
        leng = (max_ - min_) * 1.0
        #print "mean:", mean, "std:", std
        #print "left:", min_, "right:", max_
        #print "origin:", (micrograph.min(), micrograph.max())
        micrograph = np.clip(micrograph, min_, max_) - min_
        #print "clipped:", (micrograph.min(), micrograph.max())
        #print "leng:", leng, "min_:", min_
        #micrograph = micrograph / leng * 255.
        micrograph = micrograph / leng
        #micrograph = np.floor(micrograph)
        #print micrograph.min(), micrograph.max()
        #print micrograph[0:4, 0:4]


        # low pass the micrograph
        #micrograph_lowpass = scipy.ndimage.filters.gaussian_filter(micrograph, 0.1) 
        #f = np.fft.fft2(micrograph)
        #fshift = np.fft.fftshift(f)
        #magnitude_spectrum = 20*np.log(np.abs(fshift))

        #plt.subplot(121),plt.imshow(micrograph, cmap = 'gray')
        #plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        #plt.subplot(122),plt.imshow(micrograph_lowpass, cmap = 'gray')
        #plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        #plt.show() 

        # nomalize the patch
        max_value = micrograph.max()
        min_value = micrograph.min()
        #particle = (micrograph - min_value)/(max_value - min_value)
        #mean_value = micrograph.mean()
        #std_value = micrograph.std()
        #micrograph = (micrograph - mean_value)/std_value
        #micrograph = micrograph / 255.
        #
        return micrograph, pooling_size

    @staticmethod
    def preprocess_particle(particle, model_input_size):
        """Do preprocess to the particle patch after the particle data is extracted from the micrograph.

        Define this function to make sure that the same process is done to the particle 
            during the training process and picking process.
        
        Args:
            particle: numpy.array, the shape is (particle_col, particle_row)
            model_input_size: a list with length 4. The size is to fit with the model input.
                              model_input_size[0] stands for the batchsize.
                              model_input_size[1] stands for the input col.
                              model_input_size[2] stands for the input row.
                              model_input_size[3] stands for the input channel.
        Returns:
            return particle
            particle: numpy.array
        """
        # resize the particle to fit the model input
        # Here is time consuming--Huabin
        # 80-->64
        particle = scipy.misc.imresize(particle, (model_input_size[1], model_input_size[2]), interp = 'bilinear', mode = 'L')
        #particle = scipy.ndimage.zoom(particle, float(model_input_size[1])/particle.shape[1])
        # nomalize the patch
        mean_value = particle.mean()
        std_value = particle.std()
        particle = (particle - mean_value)/std_value
        # elements in particle, range[0,1]
        return particle
        
    @staticmethod
    def preprocess_particle_online(particle_batch):
        """Do process to the particle batch before they are inputed to the CNN model.

        This is online process during the training process. This process mainly includes random rotation.
        
        Args:
            particle_batch: numpy.array, the shape is (batch_size, particle_col, particle_row, channel)

        Returns:
            return particle_batch
            particle_batch: numpy.array, the shape is (batch_size, particle_col, particle_row, channel)
        """
        # random rotate the particle
        for i in range(particle_batch.shape[0]):
            random_degree = random.randint(0, 359)
            sample = particle_batch[i].reshape(particle_batch[i].shape[0], particle_batch[i].shape[1])
            max_value = sample.max()
            min_value = sample.min()
            sample = 255*(sample - min_value)/(max_value - min_value)
            sample = sample.astype('uint8')
            sample = Image.fromarray(sample)
            sample = Image.Image.rotate(sample, random_degree)
            sample = np.array(sample)
            sample = sample.reshape(particle_batch[i].shape[0], particle_batch[i].shape[1], particle_batch[i].shape[2])
            mean_value = sample.mean()
            std_value = sample.std()
            particle_batch[i] = (sample - mean_value)/std_value
        # nomalize the patch
        return particle_batch

    @staticmethod
    def read_coordinate_from_star(starfile):
        """ Read the coordinate from star file.
        return a list  
        
        Args:
            starfile: string, the input coodinate star file.
    
        Returns:
            return coordinate_list
            coordinate_list: list, the length of the list stands for the number of particles.
                             Each particle is a list too, which contains two elements.
                             The first one is the x coordinate, the second one is y coordinate.
        """
        particle_star = starRead(starfile)
        allName = particle_star.getAllName()
        if 'data_' not in allName:
            return []
        table_star = particle_star.getByName('data_')
        coordinateX_list = table_star.getByName('_rlnCoordinateX')
        coordinateY_list = table_star.getByName('_rlnCoordinateY')
        coordinate_list = []
        for i in range(len(coordinateX_list)):
            coordinate = []
            coordinate.append(int(float(coordinateX_list[i])))
            coordinate.append(int(float(coordinateY_list[i])))
            coordinate_list.append(coordinate)
        return coordinate_list

    # read the mrc file, return the header and body
    @staticmethod
    def readMrcFile(fileName):
        """Read micrograph image data from mrc format file/

        Retrieves the header information and image body information from the mrc file.
        The header is a tuple, and all the parameters about the mrc file is included in the header tuple.
        The body is a 1-d list, the data type depends on the mode parameters in header[3]
        
        Args:
            filenName: string, the input mrc file name.
    
        Returns:
            return header,body
            header: tuple, contains all the parameters in the header.
                    There are several parameters that will be used in the following operation.
                    header[0], type int, it stands for the number of columns.
                    header[1], type int, it stands for the number of rows.
                    header[2], type int, it stands for the number of sections. If the mrc file is 2-dim, then this value will be 1. 
            body: list, contains the micrograph image data information.
                  It is a 1-d list, the length is header[0]*header[1]*header[2].
                  So you can transfer it into a numpy.array and reshape it into a 2D or 3D array. 

        Raises:
            None
        """
        if not os.path.isfile(fileName):
            print("ERROR:%s is not a valid file."%(fileName))
            return

        f = open(fileName,"rb")
        data = f.read() # read all data
        f.close()
        if len(data) == 0:
            return None, None
    
        header_fmt = '10i6f3i3f2i100c3f4cifi800c'  # more information about the header please refer to mrc format in Internet.
        try:
            header = struct.unpack(header_fmt,data[0:1024])
        except:
            print "header unpack failed."
            return None, None
        n_columns = header[0]
        n_rows = header[1]
        mode = header[3]
        #print "n_columns:",n_columns
        #print "n_rows:",n_rows
        #print "mode:",mode
        if mode == 0:
            # signed 8-bit bytes range -128 to 127
            pass
        elif mode == 1:
            # 16-bit halfwords
            pass
        elif mode == 2:
            # 32-bit float
            body_fmt = str(n_columns*n_rows)+"f"
        elif mode == 3:
            # complex 16-bit integers
            pass
        elif mode == 4:
            # complex 32-bit reals
            pass
        elif mode == 6:
            # unsigned 16-bit range 0 to 65535
            pass
        else:
            print("ERROR:mode %s is not a valid value,should be [0|1|2|3|4|6]."%(fileName))
            return None
        try:
            body = list(struct.unpack(body_fmt,data[1024:]))
        except:
            print "body unpack failed"
            return None, None
        return header, body

    # read the rec file, return the header and body
    @staticmethod
    def readRecFile(fileName):
        import mrcfile
        mrc = mrcfile.open(fileName)
        header = mrc.header
        body = mrc.data
        return header, body
    
    # write numpy array to mrc file
    @staticmethod
    def writeToMrcFile(body_array, mrc_filename):
        """Write numpy 2D array to mrc format or numpy 3D array to mrcs format file/

        Store the information of numpy array into mrc format.
        The header is a tuple, and all the parameters about the mrc file is included in the header tuple.
        The body is a 1-d list, the data type depends on the mode parameters in header[3]
        
        Args:
            body_array: numpy array, 2D or 3D, type float32, 2D array refers to the micrograph and 3D array refers to the extracted particles 
            mrc_filename: string, the output mrc file name.
    
        Returns:
            None
        Raises:
            None
        """
        if body_array.dim() == 2:
            n_columns = body_array.shape()[0]
            n_rows = body_array.shape()[1]
        elif body_array.dim() == 3:
            n_section = body_array.shape()[0]
            n_columns = body_array.shape()[1]
            n_rows = body_array.shape()[2]
        else:        
            print("ERROR:the dimension of body_array must be 2 or 3")
            return

        f = open(fileName,"wb")
        data = f.read() # read all data
        f.close()
    
        header_fmt = '10i6f3i3f2i100c3f4cifi800c'  # more information about the header please refer to mrc format in Internet.
        header = struct.unpack(header_fmt,data[0:1024])
        mode = 2
        body = list(struct.unpack(body_fmt,data[1024:]))
        return header, body
    
    			
    # read particles data from star format file
    @staticmethod
    def load_Particle_From_starFile(starFileName, particle_size, model_input_size, produce_negative=True, negative_distance_ratio=0.5, negative_number_ratio=1):
        """Read the particles data from star file.

        Based on the coordinates information and the corresponding mrc file information,
        extarct the particle patch when given the particle size.
        At the same time, select some negative particles randomly.
        The coordinates of the negative particles are enforced to be away from positive particles,
        the threshold is set to negative_distance_ratio*particle_size.

        Args:
            starFileName: string, the name of the star file.
            particle_size: int, the size of the particle.
            model_input_size: the size of Placeholder to fit the model input, like [100, 64, 64, 1]
            produce_negative: bool, whether to produce the negative particles.
    			
        Returns:
            return particle_array_positive,particle_array_negative
            particle_array_positive: numpy.array, a 4-dim array,the shape is (number_particles, particle_size, particle_size, 1).
            particle_array_negative: numpy.array, a 4-dim array,the shape is (number_particles, particle_size, particle_size, 1).
     
        Raises:
            None 
        """ 
        particle_star = starRead(starFileName)
        table_star = particle_star.getByName('data_')
        mrcfilename_list = table_star.getByName('_rlnMicrographName')
        coordinateX_list = table_star.getByName('_rlnCoordinateX')
        coordinateY_list = table_star.getByName('_rlnCoordinateY')

        # creat a dictionary to store the coordinate
        # the key is the mrc file name
        # the value is a list of the coordinates
        coordinate = {}
        path_star = os.path.split(starFileName)
        for i in range(len(mrcfilename_list)):
            fileName = mrcfilename_list[i]
            fileName = os.path.join(path_star[0], fileName)
            if fileName in coordinate:
                coordinate[fileName][0].append(int(float(coordinateX_list[i])))
                coordinate[fileName][1].append(int(float(coordinateY_list[i])))
            else:
                coordinate[fileName] = [[],[]]
                coordinate[fileName][0].append(int(float(coordinateX_list[i])))
                coordinate[fileName][1].append(int(float(coordinateY_list[i])))

        # read mrc data
        particle_array_positive = []
        particle_array_negative = []
        number_total_particle = 0
        for key in coordinate:
            print key
            header, body = DataLoader.readMrcFile(key)
            n_col = header[0]
            n_row = header[1]
            body_2d = np.array(body, dtype=np.float32).reshape(n_row, n_col, 1)

            # show the micrograph with manually picked particles
            # plot the circle of the particle 
            #display.plot_circle_in_micrograph(body_2d, coordinate[key], particle_size, 'test.png') 
            # do preprocess to the micrograph
            body_2d, bin_size = DataLoader.preprocess_micrograph(body_2d)
            # bin scale the particle size and the coordinates
            particle_size_bin =int(particle_size/bin_size)
            n_col = int(n_col/bin_size)
            n_row = int(n_row/bin_size)
            for i in range(len(coordinate[key][0])):
                coordinate[key][0][i] = int(coordinate[key][0][i]/bin_size)
                coordinate[key][1][i] = int(coordinate[key][1][i]/bin_size)

            # delete the particle outside the boundry 
            radius = int(particle_size_bin/2)
            i = 0
            while True:
                if i >= len(coordinate[key][0]):
                    break

                coordinate_x = coordinate[key][0][i]
                coordinate_y = coordinate[key][1][i]
                if coordinate_x < radius or coordinate_y < radius or coordinate_y+radius > n_col or coordinate_x+radius > n_row:
                    del coordinate[key][0][i]	
                    del coordinate[key][1][i]
                else:
                    i = i + 1	

            # number of positive particles	
            number_particle = len(coordinate[key][0])
            number_total_particle = number_total_particle + number_particle
            print 'number of particles:',number_particle

            # extract the positive particles
            # store the particles in a contacted array: particle_array_positive	
            for i in range(number_particle):
                coordinate_x = coordinate[key][0][i]
                coordinate_y = coordinate[key][1][i]
                patch = np.copy(body_2d[(coordinate_y-radius):(coordinate_y+radius), (coordinate_x-radius):(coordinate_x+radius)])
                patch = DataLoader.preprocess_particle(patch, model_input_size)
                particle_array_positive.append(patch)
            # extract the negative particles
            # store the particles in a concated array: particle_array_negative	
            if produce_negative:
                for i in range(number_particle):
                    while True:
                        isLegal = True
                        coor_x = np.random.randint(radius, n_row-radius)
                        coor_y = np.random.randint(radius, n_col-radius)
                        for j in range(number_particle):
                            coordinate_x = coordinate[key][0][i]
                            coordinate_y = coordinate[key][1][i]
                            distance = ((coor_x-coordinate_x)**2+(coor_y-coordinate_y)**2)**0.5
                            if distance < negative_distance_ratio*particle_size_bin:
                                isLegal = False
                                break
                        if isLegal:
                            patch = np.copy(body_2d[(coor_y-radius):(coor_y+radius), (coor_x-radius):(coor_x+radius)])
                            patch = DataLoader.preprocess_particle(patch, model_input_size)
                            particle_array_negative.append(patch)
                            break
        if produce_negative:
	    particle_array_positive = np.array(particle_array_positive).reshape(number_total_particle, model_input_size[1], model_input_size[2], 1)	
	    particle_array_negative = np.array(particle_array_negative).reshape(number_total_particle, model_input_size[1], model_input_size[2], 1)	
   	    return particle_array_positive, particle_array_negative 

        else:
	    particle_array_positive = np.array(particle_array_positive).reshape(number_total_particle, model_input_size[1], model_input_size[2], 1)	
   	    return particle_array_positive


    @staticmethod
    def load_trainData_From_RelionStarFile(starFileName, particle_size, model_input_size, validation_ratio, train_number):    
        """read train_data and validation data from star file

        In order to train a CNN model based on Relion particle '.star' file, it need to loading the training particles
        samples from the star file.
  
        Args:
            starFileName: the name of star file
            particle_size: particle size
            model_input_size: the size of Placeholder to fit the model input, like [100, 64, 64, 1]
            validation_rate: divide the total samples into training dataset and validation dataset. 
                             This is the ratio of validation dataset compared to the total samples.

        Returns:
            return train_data,train_labels,validation_data,validation_labels
            train_data: numpy.array, np.float32, the shape is (number_samples, particle_size, particle_size, 1)  
            train_labels: numpy.array, int64, the shape is (number_samples)  
            validation_data: numpy.array, np.float32, the shape is (number_samples, particle_size, particle_size, 1)  
            validation_labels: numpy.array, int64, the shape is (number_samples)  
   
        Raises:
            None 
        """
        particle_array_positive, particle_array_negative = DataLoader.load_Particle_From_starFile(starFileName, particle_size, model_input_size)
        if train_number<len(particle_array_positive):
            particle_array_positive = particle_array_positive[:train_number, ...]
            particle_array_negative = particle_array_negative[:train_number, ...]

        np.random.shuffle(particle_array_positive)	
        np.random.shuffle(particle_array_negative)	

        validation_size = int(validation_ratio*particle_array_positive.shape[0])
        train_size = particle_array_positive.shape[0] - validation_size
        validation_data = particle_array_positive[:validation_size, ...]
        validation_data = concatenate((validation_data, particle_array_negative[:validation_size, ...]))
        validation_labels = concatenate((ones(validation_size, dtype=int64), zeros(validation_size, dtype=int64)))

        train_data = particle_array_positive[validation_size:, ...]
        train_data = concatenate((train_data, particle_array_negative[validation_size:, ...]))
        train_labels = concatenate((ones(train_size, dtype=int64), zeros(train_size, dtype=int64)))
        print train_data.shape, train_data.dtype
        print train_labels.shape, train_labels.dtype
        print validation_data.shape, validation_data.dtype
        print validation_labels.shape, validation_labels.dtype
        return train_data,train_labels, validation_data,validation_labels

    @staticmethod
    def load_Particle_From_mrcFileDir(trainInputDir, particle_size, model_input_size, coordinate_symbol, mrc_number, produce_negative = True, negative_distance_ratio = 0.5):
        """Read the particles data from mrc file dir.

        Based on the coordinates information and the corresponding mrc file information,
        extarct the particle patch when given the particle size.
        At the same time, select some negative particles randomly.
        The coordinates of the negative particles are enforced to be away from positive particles,
        the threshold is set to negative_distance_ratio*particle_size.

        Args:
            trainInputDir: string, the dir of mrc files as well as the coordinate files.
            particle_size: int, the size of the particle.
            model_input_size: the size of Placeholder to fit the model input, like [100, 64, 64, 1]
            coordinate_symbol: symbol of the coordinate file like '_manual'.
            mrc_number: number of mrc files to be used.
            produce_negative: bool, whether to produce the negative particles.
            negative_distance_ratio: float, a value between 0~1. It stands for the minimum distance between a positive sample 
                                     and negative sample compared to the particle_size. 
        Returns:
            return particle_array_positive,particle_array_negative
            particle_array_positive: numpy.array, a 4-dim array,the shape is (number_particles, particle_size, particle_size, 1).
            particle_array_negative: numpy.array, a 4-dim array,the shape is (number_particles, particle_size, particle_size, 1).
     
        Raises:
            None 
        """ 
        mrc_file_all = []
        mrc_file_coordinate = []
        file_coordinate = []
        if not os.path.isdir(trainInputDir):
            print("Invalide directory:",trainInputDir)
        
        files = os.listdir(trainInputDir)
        for f in files:
            if re.search('\.mrc$', f):
                filename = os.path.join(trainInputDir, f)
                mrc_file_all.append(filename)
        
        mrc_file_all.sort()
        for i in range(len(mrc_file_all)):
            filename_mrc = mrc_file_all[i]
            filename_coordinate = filename_mrc.replace('.mrc', coordinate_symbol+'.star') 
            if os.path.isfile(filename_coordinate):
                mrc_file_coordinate.append(filename_mrc) 
                file_coordinate.append(filename_coordinate) 

        # read mrc file 
        if mrc_number<=0:
            mrc_number = len(mrc_file_coordinate)
        else:
            if mrc_number>len(mrc_file_coordinate):
                mrc_number = len(mrc_file_coordinate)
        
        particle_array_positive = []
        particle_array_negative = []
        number_total_particle = 0
        for i in range(mrc_number):
            # read mrc data
            print(mrc_file_coordinate[i])
            header, body = DataLoader.readMrcFile(mrc_file_coordinate[i])
            if header==None or body==None:
                continue
            n_col = header[0]
            n_row = header[1]
            body_2d = np.array(body, dtype=np.float32).reshape(n_row, n_col, 1)
            # read star coordinate
            coordinate = DataLoader.read_coordinate_from_star(file_coordinate[i])
            # show the micrograph with manually picked particles
            # plot the circle of the particle 
            #display.plot_circle_in_micrograph(body_2d, coordinate, particle_size, 'test.png') 
            # do preprocess to the micrograph
            body_2d, bin_size = DataLoader.preprocess_micrograph(body_2d)
            # bin scale the particle size and the coordinates
            particle_size_bin =int(particle_size/bin_size)
            n_col = int(n_col/bin_size)
            n_row = int(n_row/bin_size)
            for i in range(len(coordinate)):
                coordinate[i][0] = int(coordinate[i][0]/bin_size)
                coordinate[i][1] = int(coordinate[i][1]/bin_size)

            # delete the particle outside the boundry 
            radius = int(particle_size_bin/2)
            i = 0
            while True:
                if i >= len(coordinate):
                    break

                coordinate_x = coordinate[i][0]
                coordinate_y = coordinate[i][1]
                if coordinate_x < radius or coordinate_y < radius or coordinate_y+radius > n_col or coordinate_x+radius > n_row:
                    del coordinate[i]	
                else:
                    i = i + 1	

            # number of positive particles	
            number_particle = len(coordinate)
            number_total_particle = number_total_particle + number_particle
            print 'number of particles:',number_particle

            # extract the positive particles
            # store the particles in a contacted array: particle_array_positive	
            for i in range(number_particle):
                coordinate_x = coordinate[i][0]
                coordinate_y = coordinate[i][1]
                patch = np.copy(body_2d[(coordinate_y-radius):(coordinate_y+radius), (coordinate_x-radius):(coordinate_x+radius)])
                patch = DataLoader.preprocess_particle(patch, model_input_size)
                particle_array_positive.append(patch)
            # extract the negative particles
            # store the particles in a concated array: particle_array_negative	
            if produce_negative:
                for i in range(number_particle):
                    while True:
                        isLegal = True
                        coor_x = np.random.randint(radius, n_row-radius)
                        coor_y = np.random.randint(radius, n_col-radius)
                        for j in range(number_particle):
                            coordinate_x = coordinate[i][0]
                            coordinate_y = coordinate[i][1]
                            distance = ((coor_x-coordinate_x)**2+(coor_y-coordinate_y)**2)**0.5
                            if distance < negative_distance_ratio*particle_size_bin:
                                isLegal = False
                                break
                        if isLegal:
                            patch = np.copy(body_2d[(coor_y-radius):(coor_y+radius), (coor_x-radius):(coor_x+radius)])
                            patch = DataLoader.preprocess_particle(patch, model_input_size)
                            particle_array_negative.append(patch)
                            break
        if produce_negative:
            particle_array_positive = np.array(particle_array_positive).reshape(number_total_particle, model_input_size[1], model_input_size[2], 1)
            particle_array_negative = np.array(particle_array_negative).reshape(number_total_particle, model_input_size[1], model_input_size[2], 1)	
            return particle_array_positive, particle_array_negative 
        else:
            particle_array_positive = np.array(particle_array_positive).reshape(number_total_particle, model_input_size[1], model_input_size[2], 1)	
            return particle_array_positive
              
    # read input data from star format file
    @staticmethod
    def load_trainData_From_mrcFileDir(trainInputDir, particle_size, model_input_size, validation_ratio, coordinate_symbol, mrc_number, train_number):    
        """read train_data and validation data from a directory of mrc files 

        Train a CNN model based on mrc files and corresponding coordinates.
  
        Args:
            trainInputDir: the directory of mrc files
            particle_size: particle size
            model_input_size: the size of Placeholder to fit the model input, like [100, 64, 64, 1]
            validation_rate: divide the total samples into training dataset and validation dataset. 
                             This is the ratio of validation dataset compared to the total samples.
            coordinate_symbol: symbol of the coordinate file like '_manual'.
            mrc_number: number of mrc files to be used.
            train_number: number of positive particles to be used for training.

        Returns:
            return train_data,train_labels,validation_data,validation_labels
            train_data: numpy.array, np.float32, the shape is (number_samples, particle_size, particle_size, 1)  
            train_labels: numpy.array, int64, the shape is (number_samples)  
            validation_data: numpy.array, np.float32, the shape is (number_samples, particle_size, particle_size, 1)  
            validation_labels: numpy.array, int64, the shape is (number_samples)  
   
        Raises:
            None 
        """
        particle_array_positive, particle_array_negative = DataLoader.load_Particle_From_mrcFileDir(trainInputDir, particle_size, model_input_size, coordinate_symbol, mrc_number)
        if train_number<len(particle_array_positive):
            particle_array_positive = particle_array_positive[:train_number, ...]
            particle_array_negative = particle_array_negative[:train_number, ...]

        np.random.shuffle(particle_array_positive)	
        np.random.shuffle(particle_array_negative)	

        validation_size = int(validation_ratio*particle_array_positive.shape[0])
        train_size = particle_array_positive.shape[0] - validation_size
        validation_data = particle_array_positive[:validation_size, ...]
        validation_data = concatenate((validation_data, particle_array_negative[:validation_size, ...]))
        validation_labels = concatenate((ones(validation_size, dtype=int64), zeros(validation_size, dtype=int64)))

        train_data = particle_array_positive[validation_size:, ...]
        train_data = concatenate((train_data, particle_array_negative[validation_size:, ...]))
        train_labels = concatenate((ones(train_size, dtype=int64), zeros(train_size, dtype=int64)))
        print train_data.shape, train_data.dtype
        print train_labels.shape, train_labels.dtype
        print validation_data.shape, validation_data.dtype
        print validation_labels.shape, validation_labels.dtype
        return train_data, train_labels, validation_data, validation_labels

    @staticmethod
    def extractData(trainInputDir, particle_size, coordinate_symbol, mrc_number, output_dir, class_number, produce_negative = True, negative_distance_ratio = 0.5):
        """extract the particles data from mrc file dir into a file.

        Based on the coordinates information and the corresponding mrc file information,
        extarct the particle patch when given the particle size.
        At the same time, select some negative particles randomly.
        The coordinates of the negative particles are enforced to be away from positive particles,
        the threshold is set to negative_distance_ratio*particle_size.
        Finally, store the extarcted particles into a file based on the 'pickle' module.
        Before writing to the file, the particles are stored in a list of length 2.
        The first element is a list of the positive particle. 
        Each element in the list of the positive particle is a numpy array, the shape is [particle_size, particle_size, 1]
        The second element is a list of the negative particle. 
        Each element in the list of the negative particle is a numpy array, the shape is [particle_size, particle_size, 1]

        Args:
            trainInputDir: string, the dir of mrc files as well as the coordinate files.
            particle_size: int, the size of the particle.
            coordinate_symbol: symbol of the coordinate file like '_manual'.
            mrc_number: number of mrc files to be used.
            produce_negative: bool, whether to produce the negative particles.
            negative_distance_ratio: float, a value between 0~1. It stands for the minimum distance between a positive sample 
                                     and negative sample compared to the particle_size. 
            output_filename: string, the file to store the particles. 
     
        Raises:
            None 
        """ 
        #TODO:class_number passed in
        output_for_each_class = [[] for i in range(class_number)] #container for each class
        symbol_all = [coordinate_symbol+str(i) for i in range(1, class_number+1)]
        mrc2coor = []

        mrc_file_all = []
        mrc_file_coordinate = []
        file_coordinate = []
        if not os.path.isdir(trainInputDir):
            print("Invalide directory:",trainInputDir)
        
        files = os.listdir(trainInputDir)
        for f in files:
            if re.search('\.mrc$', f):
                filename = os.path.join(trainInputDir, f)
                mrc_file_all.append(filename)
        
        mrc_file_all.sort()
        for i in range(len(mrc_file_all)):
            filename_mrc = mrc_file_all[i]
            file_coordinate_list = []
            for symbol in symbol_all:
                filename_coordinate = filename_mrc.replace('.mrc', symbol+'.star') 
                if os.path.isfile(filename_coordinate):
                    file_coordinate_list.append(filename_coordinate) 
            if len(file_coordinate_list) > 0:
                mrc2coor.append(file_coordinate_list) #NOTE: one mrc -> many class symbol
                mrc_file_coordinate.append(filename_mrc) 
            #filename_coordinate = filename_mrc.replace('.mrc', coordinate_symbol+'.star') 
            #if os.path.isfile(filename_coordinate):
            #    mrc_file_coordinate.append(filename_mrc) 
            #    file_coordinate.append(filename_coordinate) 

        # read mrc file 
        if mrc_number<=0:
            mrc_number = len(mrc_file_coordinate)
        
        if mrc_number>len(mrc_file_coordinate):
            mrc_number = len(mrc_file_coordinate)

        particle_array_positive = []
        particle_array_negative = []
        particle_array_positive_all_class = [[] for i in range(class_number+1)] #NOTE:1~50
        particle_array_negative_all_class = [[] for i in range(class_number+1)]
        produce_negstar_only = False
        for i in range(mrc_number):
            number_total_particle = 0
            # read mrc data
            print("%d/%d"%(i,mrc_number), mrc_file_coordinate[i])
            if produce_negstar_only:
                n_col_ori = 3838
                n_row_ori = 3710
                bin_size = 1
            else:
                header, body = DataLoader.readMrcFile(mrc_file_coordinate[i])
                #print np.max(body), np.min(body)
                if header==None or body==None:
                    continue
                n_col_ori = header[0]
                n_row_ori = header[1]
                body_2d = np.array(body, dtype=np.float32).reshape(n_row_ori, n_col_ori, 1)
                body_2d, bin_size = DataLoader.preprocess_micrograph(body_2d)
            #NOTE: read one mrc, and read many class
            # read star coordinate
            for coor_file in mrc2coor[i]:
                class_idx = int(coor_file[coor_file.rindex('class')+5: -5])
                coordinate = DataLoader.read_coordinate_from_star(coor_file)
                #print body_2d[0][0:100]
                # show the micrograph with manually picked particles
                # plot the circle of the particle 
                #coordinate = np.asarray(coordinate) / bin_size
                #coordinate = list(coordinate)
                # do preprocess to the micrograph
                # bin scale the particle size and the coordinates
                particle_size_bin =int(particle_size/bin_size)
                n_col = int(n_col_ori/bin_size)
                n_row = int(n_row_ori/bin_size)
                for j in range(len(coordinate)):
                    coordinate[j][0] = int(coordinate[j][0]/bin_size)
                    coordinate[j][1] = int(coordinate[j][1]/bin_size)

                #if i < 10:
                #    display.plot_circle_in_micrograph(body_2d, coordinate, particle_size/bin_size, 'test%d.png'%i) 
                #else:
                #    break

                # delete the particle outside the boundry 
                radius = int(particle_size_bin/2)
                k = 0
                while True:
                    if k >= len(coordinate):
                        break

                    coordinate_x = coordinate[k][0]
                    coordinate_y = coordinate[k][1]
                    if coordinate_x < radius or coordinate_y < radius or coordinate_y+radius > n_col or coordinate_x+radius > n_row:
                        del coordinate[k]
                    else:
                        k = k + 1

                # number of positive particles	
                number_particle = len(coordinate)
                number_total_particle = number_total_particle + number_particle
                #print '>>>class:%d, number of particles:%d' % (class_idx, number_particle)

                # extract the positive particles
                # store the particles in a contacted array: particle_array_positive	
                if not produce_negstar_only:
                    for j in range(number_particle):
                        coordinate_x = coordinate[j][0]
                        coordinate_y = coordinate[j][1]
                        patch = np.copy(body_2d[(coordinate_y-radius):(coordinate_y+radius), (coordinate_x-radius):(coordinate_x+radius)])
                        #patch = DataLoader.preprocess_particle(patch, model_input_size)
                        if patch.shape[0] == radius*2 and patch.shape[1] == radius*2:
                            assert (patch.shape[0] == radius*2 and patch.shape[1] == radius*2)
                            #print (">>>", patch.shape)
                            #particle_array_positive.append(patch)
                            particle_array_positive_all_class[class_idx].append(patch)
                # extract the negative particles
                # store the particles in a concated array: particle_array_negative	
                if produce_negstar_only:
                    tokens = file_coordinate[i].split('/')
                    neg_starfile = tokens[-1]
                    neg_dir = '/'.join(tokens[:-1] + ['negstar'])
                    if not os.path.exists(neg_dir):
                        os.makedirs(neg_dir)
                    neg_starfile = os.path.join(neg_dir, neg_starfile) + '.neg'
                    print (neg_starfile)
                    fout_neg = open(neg_starfile, 'w')
                    fout_neg.write('data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnClassNumber #3\n_rlnAnglePsi #4\n_rlnAutopickFigureOfMerit #5\n')
                if produce_negative:
                    for _ in range(number_particle):
                        while True:
                            isLegal = True
                            coor_x = np.random.randint(radius, n_row-radius-2)
                            coor_y = np.random.randint(radius, n_col-radius-2)
                            for ii in range(number_particle):
                                coordinate_x = coordinate[ii][0]
                                coordinate_y = coordinate[ii][1]
                                distance = ((coor_x-coordinate_x)**2+(coor_y-coordinate_y)**2)**0.5
                                if distance < negative_distance_ratio*particle_size_bin:
                                    isLegal = False
                                    break
                            if isLegal:
                                if produce_negstar_only:
                                    fout_neg.write(str(coor_x) + ' ' + str(coor_y) + ' -999 -999 -999\n')
                                    break
                                else:
                                    patch = np.copy(body_2d[(coor_y-radius):(coor_y+radius), (coor_x-radius):(coor_x+radius)])
                                    if patch.shape[0] == radius*2 and patch.shape[1] == radius*2:
                                        #patch = DataLoader.preprocess_particle(patch, model_input_size)
                                        #particle_array_negative.append(patch)
                                        particle_array_negative_all_class[class_idx].append(patch)
                                        break
                if produce_negstar_only:
                    fout_neg.close()
            print ("number_total_particle:", number_total_particle)
        # save the extracted particles into file
        #display.show_particle(np.asarray(particle_array_positive), "particle.png")
        if produce_negstar_only:
            return
        for class_idx in range(1, class_number+1):
            print (">>>>>>>>>>class_idx =", class_idx)
            output_filename = os.path.join(output_dir, "class%d.pickle"%class_idx)
            if produce_negative:
                particle = []
                print ("positive shape:", np.array(particle_array_positive_all_class[class_idx]).shape)
                print ("negative shape:", np.array(particle_array_negative_all_class[class_idx]).shape)
                size = min(len(particle_array_positive_all_class[class_idx]), len(particle_array_negative_all_class[class_idx]))
                particle_array_positive_all_class[class_idx] = particle_array_positive_all_class[class_idx][:size]
                particle_array_negative_all_class[class_idx] = particle_array_negative_all_class[class_idx][:size]
                print ("positive shape:", np.array(particle_array_positive_all_class[class_idx]).shape)
                print ("negative shape:", np.array(particle_array_negative_all_class[class_idx]).shape)
                particle.append(particle_array_positive_all_class[class_idx])
                particle.append(particle_array_negative_all_class[class_idx])
                particle = np.array(particle)
                print ("dumped particle shape=", particle.shape)
                with open(output_filename, 'wb') as f:
                    pickle.dump(particle, f)
            else:
                particle = []
                particle.append(particle_array_positive_all_class[class_idx])
                with open(output_filename, 'wb') as f:
                    pickle.dump(particle, f)

    @staticmethod
    def load_trainData_From_ExtractedDataFile(train_inputDir, train_inputFile, model_input_size, validation_ratio, train_number):
        """read train_data and validation data from pre-extracted particles.

        Train a CNN model based on pre-extracted samples. This is the cross-molecule training strategy, through which you can get a more robust CNN model to achieve better fully automated picking results.
  
        Args:
            trainInputDir: the directory of the extarcted data.
            train_inputFile: the input extarcted data file, like 'gammas.pickle;trpv1.pickle', the separator must be ';'.
            model_input_size: the size of Placeholder to fit the model input, like [100, 64, 64, 1]
            validation_rate: divide the total samples into training dataset and validation dataset. 
                             This is the ratio of validation dataset compared to the total samples.
            train_number: the number of the total positive samples. If the number is set to 10000, and there are two kinds of molecule, then each one contributes only 5,000 positive samples.  
        Returns:
            return train_data,train_labels,validation_data,validation_labels
            train_data: numpy.array, np.float32, the shape is (number_samples, particle_size, particle_size, 1)  
            train_labels: numpy.array, int64, the shape is (number_samples)  
            validation_data: numpy.array, np.float32, the shape is (number_samples, particle_size, particle_size, 1)  
            validation_labels: numpy.array, int64, the shape is (number_samples)  
   
        Raises:
            None 
        """
        input_file_list = train_inputFile.split(";")
        # define the training number of each molecule
        if train_number<=0:
            number_each_molecule = -1
        else:
            number_each_molecule = train_number//len(input_file_list)
        
        particle_array_positive = []
        particle_array_negative = []
        import time
        time1 = time.time()
        for i in range(len(input_file_list)):
            input_file_name = input_file_list[i].strip()
            input_file_name = os.path.join(train_inputDir, input_file_name)
            with open(input_file_name, 'rb') as f:
                coordinate = pickle.load(f)
                coordinate = np.expand_dims(np.array(coordinate), axis=-1)
                print (input_file_name, ">>>", np.array(coordinate).shape)
            if number_each_molecule <=0:
                number_particle = len(coordinate[0])
            else:
                if number_each_molecule > len(coordinate[0]):
                    number_particle = len(coordinate[0])
                else:
                    number_particle = number_each_molecule
                '''
                number_particle = len(coordinate[0])
                '''
            print "num particle = ", number_particle
            
            for j in range(number_particle):
                #patch_positive = DataLoader.preprocess_particle(coordinate[0][j], model_input_size)
                patch_positive = coordinate[0][j]
                particle_array_positive.append(patch_positive)
                #patch_negative = DataLoader.preprocess_particle(coordinate[1][j], model_input_size)
                patch_negative = coordinate[1][j]
                particle_array_negative.append(patch_negative)

        time2 = time.time()
        print "load time = ", (time2 - time1)
             
        number_total_particle = len(particle_array_positive)
        #particle_array_positive = np.expand_dims(np.array(particle_array_positive), axis=-1)
        #particle_array_negative = np.expand_dims(np.array(particle_array_negative), axis=-1)
        #print ">>>>>", np.array(particle_array_positive).shape
        #print ">>>>>", np.array(particle_array_negative).shape
        #particle_array_positive = np.array(particle_array_positive).reshape(number_total_particle, model_input_size[1], model_input_size[2], 1)	
        #particle_array_negative = np.array(particle_array_negative).reshape(number_total_particle, model_input_size[1], model_input_size[2], 1)	
        #particle_array_positive = np.array(particle_array_positive).reshape(number_total_particle, model_input_size[1], model_input_size[2], 1)	
        #particle_array_negative = np.array(particle_array_negative).reshape(number_total_particle, model_input_size[1], model_input_size[2], 1)	
        #np.random.shuffle(particle_array_positive)	
        #np.random.shuffle(particle_array_negative)	

        validation_size = int(validation_ratio*len(particle_array_positive))
        train_size = len(particle_array_positive) - validation_size
        validation_data = particle_array_positive[:validation_size]
        #print (validation_data.shape, particle_array_negative[:validation_size, ...].shape)
        #validation_data = concatenate((validation_data, particle_array_negative[:validation_size, ...]))
        #validation_labels = concatenate((ones(validation_size, dtype=int64), zeros(validation_size, dtype=int64)))
        validation_data = validation_data + particle_array_negative[:validation_size]
        validation_labels = concatenate((ones(validation_size, dtype=int64), zeros(validation_size, dtype=int64))).tolist()
        #validation_labels = np.array([[0.0, 1.0] for _ in range(validation_size)] + [[1.0, 0.0] for _ in range(validation_size)])

        train_data = particle_array_positive[validation_size:]
        train_data = train_data + particle_array_negative[validation_size:]
        train_labels = concatenate((ones(train_size, dtype=int64), zeros(train_size, dtype=int64))).tolist()
        #train_labels = np.array([[0.0, 1.0] for _ in range(train_size)] + [[1.0, 0.0] for _ in range(train_size)])
        print "train_data:", len(train_data)
        print "train_labels:", len(train_labels)
        print "validation_data:", len(validation_data)
        print "validation_labels:", len(validation_labels)
        return train_data, train_labels, validation_data, validation_labels

    @staticmethod
    def load_trainData_From_Class2dStarFile(train_inputDir, train_inputFile, model_input_size, validation_ratio, train_number):
        """read train_data and validation data from pre-extracted particles.

        Train a CNN model based on pre-extracted samples. This is the cross-molecule training strategy, through which you can get a more robust CNN model to achieve better fully automated picking results.
  
        Args:
            trainInputDir: the directory of the extarcted data.
            train_inputFile: the input extarcted data file, like 'gammas.pickle;trpv1.pickle', the separator must be ';'.
            model_input_size: the size of Placeholder to fit the model input, like [100, 64, 64, 1]
            validation_rate: divide the total samples into training dataset and validation dataset. 
                             This is the ratio of validation dataset compared to the total samples.
            train_number: the number of the total positive samples. If the number is set to 10000, and there are two kinds of molecule, then each one contributes only 5,000 positive samples.  
        Returns:
            return train_data,train_labels,validation_data,validation_labels
            train_data: numpy.array, np.float32, the shape is (number_samples, particle_size, particle_size, 1)  
            train_labels: numpy.array, int64, the shape is (number_samples)  
            validation_data: numpy.array, np.float32, the shape is (number_samples, particle_size, particle_size, 1)  
            validation_labels: numpy.array, int64, the shape is (number_samples)  
   
        Raises:
            None 
        """
        input_file_list = train_inputFile.split(";")
        # define the training number of each molecule
        if train_number<=0:
            number_each_molecule = -1
        else:
            number_each_molecule = train_number//len(input_file_list)
        
        particle_array_positive = []
        particle_array_negative = []
        import time
        time1 = time.time()
        file_idx = 0
        min_number = 10000000
        for i in range(len(input_file_list)):
            file_idx += 1
            input_file_name = input_file_list[i].strip()
            input_file_name = os.path.join(train_inputDir, input_file_name)
            with open(input_file_name, 'rb') as f:
                coordinate = pickle.load(f)
            print "len of coordinate 0:", len(coordinate[0])
            min_number = min(min_number, len(coordinate[0]))
            #break
            if number_each_molecule <=0:
                number_particle = len(coordinate[0])
            else:
                if number_each_molecule > len(coordinate[0]):
                    number_particle = len(coordinate[0])
                else:
                    number_particle = number_each_molecule
            print "num particle = ", number_particle
            
            if file_idx == 1:
                for j in range(number_particle):
                    patch_positive = DataLoader.preprocess_particle(coordinate[0][j], model_input_size)
                    particle_array_positive.append(patch_positive)
            elif file_idx == 2:
                for j in range(number_particle):
                    patch_negative = DataLoader.preprocess_particle(coordinate[0][j], model_input_size)
                    particle_array_negative.append(patch_negative)

        time2 = time.time()
        print "load time = ", (time2 - time1)

        particle_array_positive = particle_array_positive[:min_number]
        particle_array_negative = particle_array_negative[:min_number]
             
        number_total_particle = len(particle_array_positive)
        particle_array_positive = np.array(particle_array_positive).reshape(number_total_particle, model_input_size[1], model_input_size[2], 1)	
        particle_array_negative = np.array(particle_array_negative).reshape(number_total_particle, model_input_size[1], model_input_size[2], 1)	
        np.random.shuffle(particle_array_positive)	
        np.random.shuffle(particle_array_negative)	

        validation_size = int(validation_ratio*particle_array_positive.shape[0])
        train_size = particle_array_positive.shape[0] - validation_size
        validation_data = particle_array_positive[:validation_size, ...]
        validation_data = concatenate((validation_data, particle_array_negative[:validation_size, ...]))
        validation_labels = concatenate((ones(validation_size, dtype=int64), zeros(validation_size, dtype=int64)))

        train_data = particle_array_positive[validation_size:, ...]
        train_data = concatenate((train_data, particle_array_negative[validation_size:, ...]))
        train_labels = concatenate((ones(train_size, dtype=int64), zeros(train_size, dtype=int64)))
        print train_data.shape, train_data.dtype
        print train_labels.shape, train_labels.dtype
        print validation_data.shape, validation_data.dtype
        print validation_labels.shape, validation_labels.dtype
        return train_data, train_labels, validation_data, validation_labels

    @staticmethod
    def load_trainData_From_AutoClass2dStarFile(train_inputDir, train_inputFile, model_input_size, validation_ratio, train_number, partition, pos_list="", neg_list=""):
        #TODO:extracted_dir = 'extracted_data_recenter'
        extracted_dir = train_inputDir
        
        #auto_class_file = './relion_zw18/Class2D/2dc_2/filtered_class.txt'
        #TODO:auto_class_file = './allclass_recenter/sorted_class.txt'
        '''
        OUTPUT_dir = "/".join(train_inputDir.split("/")[:-1])
        auto_class_file = OUTPUT_dir + '/sorted_class.txt'
        finclass = open(auto_class_file, 'r')
        all_class = map(int, finclass.readline().split(','))
        print all_class
        print "#all_class =", len(all_class)
        class_num = len(all_class)
        #partition = int(class_num / 2)
        #partition = 8
        #good_class = all_class[:partition]
        #bad_class  = all_class[partition:]
        '''
        pos_list = pos_list.strip()
        neg_list = neg_list.strip()
        print "pos_list:", pos_list
        print "neg_list:", neg_list
        good_class = map(int, pos_list.split(','))
        if neg_list == "":
            bad_class = []
        else:
            bad_class = map(int, neg_list.split(','))
        assert(len(good_class) > 0)
        #assert(len(bad_class) > 0)
        print ("good_class:", good_class)
        print ("bad_class:", bad_class)
        sys.stdout.flush()

        for p in good_class:
            assert(p not in bad_class)
        for n in bad_class:
            assert(n not in good_class)

        #TODO: if extract_dir is empty?
        for root, dirs, files in os.walk(extracted_dir):
            input_file_list = [x for x in files if x.find('class') != -1]
            break
        assert(len(input_file_list) > 0)
        print input_file_list
        #input_file_list = train_inputFile.split(";")
        # define the training number of each molecule
        if train_number<=0:
            number_each_molecule = -1
        else:
            number_each_molecule = train_number//len(input_file_list)
        
        particle_array_positive = []
        particle_array_negative = []
        import time, re
        time1 = time.time()
        for i in range(len(input_file_list)):
            file_class = re.findall('class(\d+).pickle', input_file_list[i].split('/')[-1])
            file_class = int(file_class[0])
            input_file_name = input_file_list[i].strip()
            input_file_name = os.path.join(train_inputDir, input_file_name)
            with open(input_file_name, 'rb') as f:
                coordinate = pickle.load(f)
            #break
            number_particle_pos = len(coordinate[0])
            number_particle_neg = len(coordinate[1])
            print "#star of %s:"%input_file_name, number_particle_pos, file_class
            
            if file_class in good_class:
                for j in range(number_particle_pos):
                    patch_positive = DataLoader.preprocess_particle(coordinate[0][j], model_input_size)
                    particle_array_positive.append(patch_positive)
                for j in range(number_particle_neg):
                    patch_negative = DataLoader.preprocess_particle(coordinate[1][j], model_input_size)
                    particle_array_negative.append(patch_negative)
            elif file_class in bad_class:
                for j in range(number_particle_pos):
                    patch_negative = DataLoader.preprocess_particle(coordinate[0][j], model_input_size)
                    particle_array_negative.append(patch_negative)

        time2 = time.time()
        #print "load time = ", (time2 - time1)

        posi_len = len(particle_array_positive)
        nega_len = len(particle_array_negative)
        min_number = min(posi_len, nega_len)
        print "posi:%d, nega:%d" % (posi_len, nega_len)
        particle_array_positive = particle_array_positive[:min_number]
        particle_array_negative = particle_array_negative[:min_number]

        posi_len = len(particle_array_positive)
        nega_len = len(particle_array_negative)
        print "posi:%d, nega:%d" % (posi_len, nega_len)
        sys.stdout.flush()

        number_total_particle = len(particle_array_positive)
        particle_array_positive = np.array(particle_array_positive).reshape(number_total_particle, model_input_size[1], model_input_size[2], 1)	
        particle_array_negative = np.array(particle_array_negative).reshape(number_total_particle, model_input_size[1], model_input_size[2], 1)	
        np.random.shuffle(particle_array_positive)	
        np.random.shuffle(particle_array_negative)	

        validation_size = int(validation_ratio*particle_array_positive.shape[0])
        train_size = particle_array_positive.shape[0] - validation_size
        validation_data = particle_array_positive[:validation_size, ...]
        validation_data = concatenate((validation_data, particle_array_negative[:validation_size, ...]))
        validation_labels = concatenate((ones(validation_size, dtype=int64), zeros(validation_size, dtype=int64)))

        train_data = particle_array_positive[validation_size:, ...]
        train_data = concatenate((train_data, particle_array_negative[validation_size:, ...]))
        train_labels = concatenate((ones(train_size, dtype=int64), zeros(train_size, dtype=int64)))
        #train_labels = concatenate(([[0.0, 1.0] for _ in range(train_size)], [[1.0, 0.0] for _ in range(train_size)]), axis=0)
        #train_labels = np.array([[0.0, 1.0] for _ in range(train_size)] + [[1.0, 0.0] for _ in range(train_size)])
        print train_data.shape, train_data.dtype
        print train_labels.shape, train_labels.dtype
        print validation_data.shape, validation_data.dtype
        print validation_labels.shape, validation_labels.dtype
        sys.stdout.flush()
        return train_data, train_labels, validation_data, validation_labels

    @staticmethod
    def load_trainData_From_PrePickedResults(train_inputDir, train_inputFile, particle_size, model_input_size, validation_ratio, train_number):
        """read train_data and validation data from pre-picked results

        Train a CNN model based on pre-picked particles. Then you can pick the particles again based on the new trained model.
        This will improve the precision and recall of picking results.
  
        Args:
            trainInputDir: the directory of mrc files
            trainInputFile: the file of the pre-picked results, like '/Your_pick_path/autopick_results.pickle'
            particle_size: particle size
            model_input_size: the size of Placeholder to fit the model input, like [100, 64, 64, 1]
            validation_rate: divide the total samples into training dataset and validation dataset. 
                             This is the ratio of validation dataset compared to the total samples.
            train_number: if the value is ranging (0,1), then it means the prediction threshold. If the value is ranging (1,100), then it means the proportion of top sorted ranking particles. If the value is larger than 100, then it means the number of top sorted ranking particles.

        Returns:
            return train_data,train_labels,validation_data,validation_labels
            train_data: numpy.array, np.float32, the shape is (number_samples, particle_size, particle_size, 1)  
            train_labels: numpy.array, int64, the shape is (number_samples)  
            validation_data: numpy.array, np.float32, the shape is (number_samples, particle_size, particle_size, 1)  
            validation_labels: numpy.array, int64, the shape is (number_samples)  
   
        Raises:
            None 
        """
        with open(train_inputFile, 'rb') as f:
            coordinate = pickle.load(f)
            """
            coordinate: a list, the length of it stands for the number of picked micrograph file.
                        Each element is a list too, which contains all coordinates from the same micrograph. 
                        The length of the list stands for the number of the particles.
                        And each element in the list is a small list of length of 4.
                        The first element in the small list is the coordinate x-aixs. 
                        The second element in the small list is the coordinate y-aixs. 
                        The third element in the small list is the prediction score. 
                        The fourth element in the small list is the micrograh name. 
            """
        # sort all particles based on the prediction score
        # get the top ranked particles
        if train_number>1:
            train_number = int(train_number)
            particles_all = []
            for i in range(len(coordinate)):
                for j in range(len(coordinate[i])):
                    particles_all.append(coordinate[i][j])
            
            # sort all particles based on prediction score in descending order
            particles_all = sorted(particles_all, key=itemgetter(2), reverse=True)
            if train_number < 100 :
                number_positive_samples = len(particles_all)*train_number/100
            else:
                number_positive_samples = train_number
             
            print ("number_positive_samples:",number_positive_samples)
            particles_train = particles_all[:number_positive_samples]
             
            # recover 'particles_train' to the formate like 'coordinate'
            particles_train = sorted(particles_train, key=itemgetter(3)) 
            mrc_filename = particles_train[0][3]
            coordinate = []
            mrc_coordinate = []
            for i in range(len(particles_train)):
                if particles_train[i][3]==mrc_filename:
                    mrc_coordinate.append(particles_train[i])
                else:
                    coordinate.append(mrc_coordinate)
                    mrc_coordinate = []
                    mrc_filename = particles_train[i][3]
                    mrc_coordinate.append(particles_train[i])
                if i==len(particles_train)-1:
                    coordinate.append(mrc_coordinate)

        # read mrc data
        particle_array_positive = []
        particle_array_negative = []
        number_total_particle = 0
        negative_distance_ratio = 0.5
        for i in range(len(coordinate)):
            mrc_filename = coordinate[i][0][3]
            #print(mrc_filename)
            mrc_filename = os.path.basename(mrc_filename)
            mrc_filename = os.path.join(train_inputDir, mrc_filename)
            print(mrc_filename)
            header,body = DataLoader.readMrcFile(mrc_filename)
            n_col = header[0]
            n_row = header[1]
            body_2d = np.array(body, dtype=np.float32).reshape(n_row, n_col, 1)

            # show the micrograph with manually picked particles
            # plot the circle of the particle 
            #display.plot_circle_in_micrograph(body_2d, coordinate[key], particle_size, 'test.png') 
            # do preprocess to the micrograph
            body_2d, bin_size = DataLoader.preprocess_micrograph(body_2d)
            # bin scale the particle size and the coordinates
            particle_size_bin =int(particle_size/bin_size)
            radius = int(particle_size_bin/2)
            n_col = int(n_col/bin_size)
            n_row = int(n_row/bin_size)
            for j in range(len(coordinate[i])):
                coordinate[i][j][0] = int(coordinate[i][j][0]/bin_size)
                coordinate[i][j][1] = int(coordinate[i][j][1]/bin_size)

            if train_number>0 and train_number<1:
                coordinate_positive = []
                for j in range(len(coordinate[i])):
                    if coordinate[i][j][2]>train_number:
                        coordinate_positive.append(coordinate[i][j])
            else:
                coordinate_positive = coordinate[i]

            # number of positive particles      
            number_particle = len(coordinate_positive)
            number_total_particle = number_total_particle + number_particle
            print 'number of particles:',number_particle

            # extract the positive particles
            # store the particles in a contacted array: particle_array_positive 
            for j in range(number_particle):
                coordinate_x = coordinate_positive[j][0]
                coordinate_y = coordinate_positive[j][1]
                patch = np.copy(body_2d[(coordinate_y-radius):(coordinate_y+radius), (coordinate_x-radius):(coordinate_x+radius)])
                patch = DataLoader.preprocess_particle(patch, model_input_size)
                particle_array_positive.append(patch)
            # extract the negative particles
            # store the particles in a concated array: particle_array_negative  
            for i in range(number_particle):
                while True:
                    isLegal = True
                    coor_x = np.random.randint(radius, n_row-radius)
                    coor_y = np.random.randint(radius, n_col-radius)
                    for j in range(number_particle):
                        coordinate_x = coordinate_positive[j][0]
                        coordinate_y = coordinate_positive[j][1]
                        distance = ((coor_x-coordinate_x)**2+(coor_y-coordinate_y)**2)**0.5
                        if distance < negative_distance_ratio*particle_size_bin:
                            isLegal = False
                            break
                    if isLegal:
                        patch = np.copy(body_2d[(coor_y-radius):(coor_y+radius), (coor_x-radius):(coor_x+radius)])
                        patch = DataLoader.preprocess_particle(patch, model_input_size)
                        particle_array_negative.append(patch)
                        break
        
        # reshape all the positive samples and negative samples    
        particle_array_positive = np.array(particle_array_positive).reshape(number_total_particle, model_input_size[1], model_input_size[2], 1)
        particle_array_negative = np.array(particle_array_negative).reshape(number_total_particle, model_input_size[1], model_input_size[2], 1)
        np.random.shuffle(particle_array_positive)	
        np.random.shuffle(particle_array_negative)	

        validation_size = int(validation_ratio*particle_array_positive.shape[0])
        train_size = particle_array_positive.shape[0] - validation_size
        validation_data = particle_array_positive[:validation_size, ...]
        validation_data = concatenate((validation_data, particle_array_negative[:validation_size, ...]))
        validation_labels = concatenate((ones(validation_size, dtype=int64), zeros(validation_size, dtype=int64)))

        train_data = particle_array_positive[validation_size:, ...]
        train_data = concatenate((train_data, particle_array_negative[validation_size:, ...]))
        train_labels = concatenate((ones(train_size, dtype=int64), zeros(train_size, dtype=int64)))
        print train_data.shape, train_data.dtype
        print train_labels.shape, train_labels.dtype
        print validation_data.shape, validation_data.dtype
        print validation_labels.shape, validation_labels.dtype
        return train_data, train_labels, validation_data, validation_labels

