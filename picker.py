#import logging
#logging.getLogger("tensorflow").setLevel(logging.WARNING)
import os, time, re, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from deepModel import DeepModel
from dataLoader import DataLoader 
from optparse import OptionParser
import display
import pickle
from operator import itemgetter
import copy
import sys

class Picker:
    def __init__(self):
        parser = OptionParser()
        parser.add_option("--inputDir", dest="inputDir", 
                help="Input directory", metavar="DIRECTORY")
        parser.add_option("--outputDir", dest="outputDir", 
                help="Output directory, the coordinates file will be saved here.", metavar="DIRECTORY")
        parser.add_option("--coorOutput", dest="coorOutput", 
                help="Output directory, the coordinates file will be saved here.", metavar="DIRECTORY", default="pick-result")
        parser.add_option("--pre_trained_model", dest="pre_trained_model", 
                help="Input the pre-trained model", metavar="FILE")
        parser.add_option("--mrc_number", dest="mrc_number", 
                help="Number of mrc files to be picked.", metavar="VALUE", default=-1)
        parser.add_option("--mrc_filename", dest="mrc_filename",
                help="mrc filename to be picker.", metavar="STRING", default="none")
        parser.add_option("--particle_size", dest="particle_size", 
                help="the size of the particle.", metavar="VALUE", default=-1)
        parser.add_option("--coordinate_symbol", dest="coordinate_symbol", 
                help="The symbol of the saveed coordinate file, like '_cnnPick'", metavar="STRING")
        parser.add_option("--threshold", dest="threshold", 
                help="Pick the particles, the prediction value is larger than the threshold..", 
                metavar="VALUE", default=0.5)
        parser.add_option("--contrast", dest="contrast",
                help="Contrast(1~5)",
                metavar="STRING", default='3')
        parser.add_option("--deeppickerRunDir", dest="deeppickerRunDir",
                help="DeepRun directory", metavar="DIRECTORY", default='./')
        parser.add_option("--gpu", dest="gpu",
                help="gpu device",
                metavar="STRING", default='0')
        parser.add_option("--edge", dest="edge",
                help="edge",
                metavar="STRING", default='1')
        (opt, args) = parser.parse_args()
        self.model_input_size = [1000, 64, 64, 1]
        self.num_class = 2
        self.batch_size = self.model_input_size[0]
        self.particle_size = int(opt.particle_size)
        self.edge = float(opt.edge)
        self.pre_trained_model = opt.pre_trained_model
        self.input_dir = opt.inputDir
        self.output_dir = opt.outputDir
        if self.input_dir[0] != '/':
            if self.input_dir == '.':
                self.input_dir = self.input_dir[1:]
            if self.input_dir[0:2] == './':
                self.input_dir = self.input_dir[2:]
            self.input_dir = os.path.join(opt.deeppickerRunDir, self.input_dir)

        if self.output_dir[0] != '/':
            if self.output_dir == '.':
                self.output_dir = self.output_dir[1:]
            if self.output_dir[0:2] == './':
                self.output_dir = self.output_dir[2:]
            self.output_dir = os.path.join(opt.deeppickerRunDir, self.output_dir)

        self.coor_output = os.path.join(self.output_dir, opt.coorOutput)
        self.threshold = float(opt.threshold)
        self.coordinate_symbol = opt.coordinate_symbol
        self.mrc_number = int(opt.mrc_number)
        self.mrc_filename = opt.mrc_filename
        self.deepModel = DeepModel(self.particle_size, self.model_input_size, self.num_class)
        self.deeppicker_conf = os.path.join(opt.deeppickerRunDir, 'deeppicker.conf')
        self.plot_picking_result = False
        self.verbose = False

        print ("INPUT_DIR>>>", self.input_dir)
        print ("OUTPUT_DIR>>>", self.output_dir)
        print ("CONFIG_FILE>>>", self.deeppicker_conf)
        print ("MODEL>>>", self.pre_trained_model)
        '''
        with open(self.deeppicker_conf, 'w') as fout:
            print>>fout, "[General]"
            print>>fout, "import_mrc=" + self.input_dir
            print>>fout, "output_path=" + self.output_dir
            print>>fout, "coor_output=" + opt.coorOutput
            print>>fout, "model_path=" + self.pre_trained_model
            print>>fout, "particle_size=" + str(self.particle_size)
            print>>fout, "threshold=" + str(self.threshold)
            print>>fout, "symbol=" + self.coordinate_symbol
            print>>fout, "class2d_name=default_job"
            print>>fout, "contrast=" + str(opt.contrast)
            print>>fout, "gpu=" + "\""+str(opt.gpu)+"\""
        '''
            
        #TODO: tuning threshold and minimum_distance_rate
        #if self.plot_picking_result:
        #    if os.path.exists('plot') == False:
        #        os.makedirs('plot')


    def pick_particle(self):
        if (not os.path.isfile(self.pre_trained_model)) and \
           (not os.path.isfile(self.pre_trained_model+'.index')):
            print("ERROR:%s is not a valid file." % (self.pre_trained_model))
            return
        input_dir = '/'.join(self.input_dir.split('/')[:-1])
        mrc_filter = self.input_dir.split('/')[-1]
        if not os.path.isdir(input_dir):
            print("ERROR:%s is not a valid dir." % (input_dir))
            return
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.isdir(self.coor_output):
            os.mkdir(self.coor_output)

        # load mrc files
        mrc_file_all = []
        files = os.listdir(input_dir)
        import glob
        mrc_file_all = glob.glob(self.input_dir)
        print ("Picking files>>>")
        total_mf = len(mrc_file_all)
        for i, mf in enumerate(mrc_file_all):
            print "%d/%d %s" % (i+1, total_mf, mf)
        #for f in files:
            #if re.search('\.mrc$', f):
            #    filename = os.path.join(input_dir, f)
            #    mrc_file_all.append(filename)

        mrc_file_all.sort()
        #mrc_number = max(0, min(self.mrc_number, len(mrc_file_all)))
        mrc_number = len(mrc_file_all)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2, allow_growth=False)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)) as self.sess:
            # load trained model
            saver = tf.train.Saver(tf.all_variables())
            saver.restore(self.sess, self.pre_trained_model)
            # do the autopick
            time1 = time.time()
            candidate_particle_all = []
            if self.mrc_filename != "none":
                # multi mrc path
                mrc_file_paths = self.mrc_filename.split(',')
                for i, mrc_filename_tmp in enumerate(mrc_file_paths):
                    time_once = time.time()
                    mrc_file_path = os.path.join(input_dir, mrc_filename_tmp)
                    print ">>>%d/%d %s" % (i+1, len(mrc_file_paths), mrc_file_path)
                    assert(os.path.exists(mrc_file_path))
                    coordinate, coordinate_all = self.pick(mrc_file_path)
                    sys.stdout.flush()
                    candidate_particle_all.append(coordinate)
                    self.write_coordinate(coordinate, mrc_file_path, self.coordinate_symbol, self.threshold, self.coor_output)
                    self.write_coordinate(coordinate_all, mrc_file_path, self.coordinate_symbol, 0.0, self.coor_output, save_all=True)
                    time_once_cost = time.time() - time_once
                    print("pick_one time: %.1f s" % time_once_cost)
            else:
                for i in range(mrc_number):
                    mrc_file = mrc_file_all[i]
                    time_once = time.time()

                    print ">>>%d/%d %s" % (i+1, mrc_number, mrc_file)
                    coordinate, coordinate_all = self.pick(mrc_file)
                    candidate_particle_all.append(coordinate)
                    self.write_coordinate(coordinate, mrc_file, self.coordinate_symbol, self.threshold, self.coor_output)
                    self.write_coordinate(coordinate_all, mrc_file, self.coordinate_symbol, 0.0, self.coor_output, save_all=True)

                    time_once_cost = time.time() - time_once
                    print("pick_one time: %.1f s" % time_once_cost)
            time_cost = time.time() - time1
            print("total time: %.1f s" % time_cost)
            # write all pick results to file
            pickle_output_file = os.path.join(self.coor_output, 'pick_results.pickle')
            self.write_all_pick_results(candidate_particle_all, pickle_output_file)

    def peak_detection(self, image_2D, local_window_size):
        #print image_2D
        #print image_2D.shape
        #print local_window_size
        col = image_2D.shape[0]
        row = image_2D.shape[1]
        print (col, row)
        data_max = filters.maximum_filter(image_2D, local_window_size)
        maxima = (image_2D == data_max)
        data_min = filters.minimum_filter(image_2D, local_window_size)
        diff = ((data_max - data_min) > 0)
        maxima[diff==0] = 0

        labeled, num_objects = ndimage.label(maxima)
        # get the coordinate of the local maximum
        # the shape of the array_y_x is (number, 2)
        array_y_x = np.array(ndimage.center_of_mass(image_2D, labeled, range(1, num_objects+1)))
        array_y_x = array_y_x.astype(int)
        list_y_x = array_y_x.tolist()
        #print("number of local maximum:%d"%len(list_y_x))
        for i in range(len(list_y_x)):
            # add the prediction score to the list
            list_y_x[i].append(image_2D[ array_y_x[i][0] ][array_y_x[i][1] ]) 
            # add a symbol to the list, and it is used to remove crowded candidate
            list_y_x[i].append(0)

        # remove close candidate
        for i in range(len(list_y_x)-1):
            if list_y_x[i][3] == 1:
                continue
            
            for j in range(i+1, len(list_y_x)):
                #if list_y_x[i][3] == 1:
                #    break
                if list_y_x[j][3] == 1:
                    continue
                d_y = list_y_x[i][0] - list_y_x[j][0]
                d_x = list_y_x[i][1] - list_y_x[j][1]
                d_distance = math.sqrt(d_y**2 + d_x**2)
                if d_distance < local_window_size/2:
                    if list_y_x[i][2] >= list_y_x[j][2]:
                        list_y_x[j][3] = 1
                    else:
                        list_y_x[i][3] = 1  

        # remove edge candidate
        particle_size_bin = local_window_size/2
        thres = max(0, self.edge)
        for i in range(len(list_y_x)):
            if list_y_x[i][3] == 1:
                continue
            #print (list_y_x[i][0], list_y_x[i][1])
            if list_y_x[i][0] < particle_size_bin * thres or list_y_x[i][0] > row - particle_size_bin * thres:
                list_y_x[i][3] = 1
            if list_y_x[i][1] < particle_size_bin * thres or list_y_x[i][1] > col - particle_size_bin * thres:
                list_y_x[i][3] = 1
                
        list_coordinate_clean = []
        for i in range(len(list_y_x)):
            if list_y_x[i][3] == 0:
                # remove the symbol element
                list_x_y = []
                list_x_y.append(list_y_x[i][1])
                list_x_y.append(list_y_x[i][0])
                list_x_y.append(list_y_x[i][2])
                list_coordinate_clean.append(list_x_y)

        return list_coordinate_clean
#
    def pick(self, mrc_filename):
        if mrc_filename.endswith('.rec'):
            header, body = DataLoader.readRecFile(mrc_filename)
        else:
            header, body = DataLoader.readMrcFile(mrc_filename)
        if header == None or body == None:
            return []
        num_col = header[0]
        num_row = header[1]
        body_2d = np.array(body, dtype=np.float32).reshape(num_row, num_col)
        body_2d_ori = body_2d

        body_2d, bin_size = DataLoader.preprocess_micrograph(body_2d)
        step_size = 4
        candidate_patches = None
        candidate_patches_exist = False
        num_total_patch = 0
        patch_size = int(self.particle_size/bin_size)
        local_window_size = int(patch_size/step_size)
        #local_window_size = int(0.6*patch_size)
        map_col = int((body_2d.shape[0]-patch_size+1)/step_size)
        map_row = int((body_2d.shape[1]-patch_size+1)/step_size)
        time1 = time.time()
        particle_candidate_all = []
        map_index_col = 0
        for col in range(0, body_2d.shape[0]-patch_size, step_size):
            for row in range(0, body_2d.shape[1]-patch_size, step_size):
                patch = np.copy(body_2d[col:(col+patch_size), row:(row+patch_size)])
                #patch = DataLoader.preprocess_particle(patch, self.model_input_size)
                particle_candidate_all.append(patch)
                num_total_patch = num_total_patch + 1
            map_index_col = map_index_col + 1
        
        map_index_row = map_index_col - map_col + map_row
        #particle_candidate_all = np.array(particle_candidate_all).reshape(
        #        num_total_patch, self.model_input_size[1], self.model_input_size[2], 1)
        particle_candidate_all = np.array(particle_candidate_all).reshape(
                num_total_patch, patch_size, patch_size, 1)
        predictions = self.deepModel.evaluation(particle_candidate_all, self.sess)
        predictions = predictions[:, 1:2]
        predictions = predictions.reshape(map_index_col, map_index_row)
        time_cost = time.time() - time1
        if self.verbose:
            print("gpu time: %.1f s" % time_cost)
        list_coordinate = self.peak_detection(predictions, local_window_size)
        for i in range(len(list_coordinate)):
            list_coordinate[i].append(mrc_filename)
            list_coordinate[i][0] = (list_coordinate[i][0]*step_size+patch_size/2) * bin_size
            list_coordinate[i][1] = (list_coordinate[i][1]*step_size+patch_size/2) * bin_size

        #return all coordinate
        list_coordinate_all = [i for i in list_coordinate if i[2] > 0.0]
        list_coordinate_all = sorted(list_coordinate_all, key=lambda x: x[2], reverse=True)

        #print ("size = ", len(list_coordinate))
        list_coordinate = [i for i in list_coordinate if i[2] > self.threshold]
        list_coordinate = sorted(list_coordinate, key=lambda x: x[2], reverse=True)
        #print ("filtered size = ", len(list_coordinate))
        #list_coordinate = list_coordinate[:100]
        print ("#candidate:%d, #picked:%d" % (num_total_patch, len(list_coordinate)) )
        plot_list_coordinate = copy.deepcopy(list_coordinate)

        for i in range(len(plot_list_coordinate)):
            plot_list_coordinate[i][0] = plot_list_coordinate[i][0] / bin_size
            plot_list_coordinate[i][1] = plot_list_coordinate[i][1] / bin_size
        if self.plot_picking_result:
            #print ">>>>>>>>>>>>>>>", body_2d.shape
            reference_coordinate_file = mrc_filename.replace('.mrc', '_DW_recentered.star')
            reference_coordinate_file = os.path.join('/data00/Data/piezo/train', reference_coordinate_file)
            #print(reference_coordinate_file)
            if os.path.isfile(reference_coordinate_file):
                reference_coordinate = DataLoader.read_coordinate_from_star(reference_coordinate_file)
                for i in range(len(reference_coordinate)):
                    reference_coordinate[i][0] = reference_coordinate[i][0] / bin_size
                    reference_coordinate[i][1] = reference_coordinate[i][1] / bin_size
            #display.plot_circle_in_micrograph(body_2d, reference_coordinate, plot_list_coordinate, patch_size, "plot/micro_circle_%s.png" % (os.path.basename(mrc_filename)))
            #plot_dir = os.path.basename(self.output_dir)
            plot_dir = os.path.join(os.path.abspath(self.output_dir), "plot")
            #pos = plot_dir.rfind('/')
            #plot_dir = os.path.join(plot_dir[:pos], 'plot')
            if self.verbose:
                print "plot_dir >>>>>>>>>> ", plot_dir
            if self.plot_picking_result and os.path.exists(plot_dir) == False:
                os.makedirs(plot_dir)
            display.plot_circle_in_micrograph(body_2d, plot_list_coordinate, patch_size, plot_dir + "/micro_circle_%s.png" % (os.path.basename(mrc_filename)))
        #display.plot_circle_in_micrograph(body_2d_ori, list_coordinate, self.particle_size, "plot/micro_circle_%s.png" % (os.path.basename(mrc_filename)))
        return list_coordinate, list_coordinate_all

    def write_coordinate(self, coordinate, mrc_filename, coordinate_symbol, threshold, output_dir, save_all=False):
        mrc_basename = os.path.basename(mrc_filename)
        #print(mrc_basename)
        coordinate_name = os.path.join(output_dir, mrc_basename[:-4]+coordinate_symbol+".star")
        if save_all:
            coordinate_name += ".all"
        print "STAR_FILE:", coordinate_name
        f = open(coordinate_name, 'w')
        f.write('data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnClassNumber #3\n_rlnAnglePsi #4\n_rlnAutopickFigureOfMerit #5\n')
        for i in range(len(coordinate)):
            if coordinate[i][2] > threshold:
                if save_all:
                    f.write(str(coordinate[i][0]) + ' ' + str(coordinate[i][1]) + ' ' + str(coordinate[i][2]) + ' -999 -999 -999\n') # + ' ' + str(coordinate[i][2]) + '\n')
                else:
                    f.write(str(coordinate[i][0]) + ' ' + str(coordinate[i][1]) + ' -999 -999 -999\n') # + ' ' + str(coordinate[i][2]) + '\n')
        f.close()

    def write_all_pick_results(self, coordinate, output_file):
        with open(output_file, 'wb') as f:
            pickle.dump(coordinate, f)

    @staticmethod
    def analysis_pick_results(pick_results_file, reference_coordinate_dir, reference_coordinate_symbol, particle_size, minimum_distance_rate):
        """Load the picking results from a file of binary format and compare it with the reference coordinate.
        This function analysis the picking results with reference coordinate and calculate the recall, precision and the deviation from the center.
        Args:
            pick_results_file: string, the file name of the pre-picked results.
            reference_mrc_dir: string, the directory of the mrc file dir.
            reference_coordinate_symbol: the symbol of the coordinate, like '_manualpick'
            particle_size: int, the size of particle
            minimum_distance_rate: float, the default is 0.2, a picked coordinate is considered to be a true positive only when the distance between the picked coordinate and the reference coordinate is less than minimum_distance_rate mutiplicate particle_size.
        """
        with open(pick_results_file, 'rb') as f:
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
        tp = 0.
        total_pick = 0
        total_reference = 0
        coordinate_total = []
        print len(coordinate)
        total_analyse_num = 0
        for i in range(len(coordinate)):
            #print coordinate[i]
            #print coordinate[i]
            if len(coordinate[i]) == 0:
                continue
            total_analyse_num += 1
            mrc_filename = os.path.basename(coordinate[i][0][3])
            #print(mrc_filename)
            reference_coordinate_file = mrc_filename.replace('.mrc', reference_coordinate_symbol+'.star')
            reference_coordinate_file = os.path.join(reference_coordinate_dir, reference_coordinate_file)
            #print(reference_coordinate_file)
            if os.path.isfile(reference_coordinate_file):
                reference_coordinate = DataLoader.read_coordinate_from_star(reference_coordinate_file)
                """
                reference_coordinate: a list, the length of it stands for the number of picked particles.
                            And each element in the list is a small list of length of 2.
                            The first element in the small list is the coordinate x-aixs. 
                            The second element in the small list is the coordinate y-aixs. 
                """    
                tp_sigle, average_distance = Picker.calculate_tp(coordinate[i], reference_coordinate, particle_size*minimum_distance_rate)
                print("tp:",tp_sigle)
                print("average_distance:",average_distance)
                # calculate the number of true positive, when the threshold is set to 0.5
                tp_sigle = 0.
                total_reference = total_reference + len(reference_coordinate)
                for j in range(len(coordinate[i])):
                    coordinate_total.append(coordinate[i][j])
                    #if coordinate[i][j][2]>minimum_distance_rate:
                    threshold = 0.99
                    if coordinate[i][j][2]>threshold:
                        total_pick = total_pick + 1
                        if coordinate[i][j][4] == 1:
                            tp = tp + 1
                            tp_sigle = tp_sigle + 1
                print(tp_sigle/len(reference_coordinate))
            else:
                print("Can not find the reference coordinate:"+reference_coordinate_file)
        print "tp=", tp
        print "total_pick=", total_pick
        print "total_analyse_num=", total_analyse_num
        precision = tp / total_pick
        recall = tp / total_reference
        print("(threshold %.2f)precision:%f recall:%f"%(minimum_distance_rate, precision, recall))
        # sort the coordinate based on prediction score in a descending order.
        coordinate_total = sorted(coordinate_total, key = itemgetter(2), reverse = True) 
        total_tp = []
        total_recall = []
        total_precision = []
        total_probability = []
        total_average_distance = []
        total_distance = 0.
        tp_tem = 0.
        for i in range(len(coordinate_total)):
            if coordinate_total[i][4] == 1:
                tp_tem = tp_tem + 1
                total_distance = total_distance + coordinate_total[i][5]
            precision = tp_tem / (i+1)
            recall = tp_tem / total_reference
            total_tp.append(tp_tem)
            total_recall.append(recall)
            total_precision.append(precision)
            total_probability.append(coordinate_total[i][2])
            if tp_tem==0:
                average_distance = 0
            else:
                average_distance = total_distance/tp_tem
            total_average_distance.append(average_distance)
        # write the list results in file
        directory_pick = os.path.dirname(pick_results_file)
        total_results_file = os.path.join(directory_pick, 'results.txt')
        f = open(total_results_file, 'w')
        # write total_tp
        f.write(','.join(map(str, total_tp))+'\n')
        f.write(','.join(map(str, total_recall))+'\n')
        f.write(','.join(map(str, total_precision))+'\n')
        f.write(','.join(map(str, total_probability))+'\n')
        f.write(','.join(map(str, total_average_distance))+'\n')
        f.write('#total autopick number:%d\n'%(len(coordinate_total))) 
        f.write('#total manual pick number:%d\n'%(total_reference))
        f.write('#the first row is number of true positive\n')
        f.write('#the second row is recall\n')
        f.write('#the third row is precision\n')
        f.write('#the fourth row is probability\n')
        f.write('#the fiveth row is distance\n')
        
        # show the recall and precision
        times_of_manual = len(coordinate_total)//total_reference + 1
        for i in range(times_of_manual):
            print('autopick_total sort, take the head number of total_manualpick * ratio %d'%(i+1))
            f.write('#autopick_total sort, take the head number of total_manualpick * ratio %d \n'%(i+1))
            if i==times_of_manual-1:
                print('precision:%f \trecall:%f'%(total_precision[-1], total_recall[-1]))
                f.write('precision:%f \trecall:%f \n'%(total_precision[-1], total_recall[-1]))
            else:
                print('precision:%f \trecall:%f'%(total_precision[(i+1)*total_reference-1], total_recall[(i+1)*total_reference-1]))
                f.write('precision:%f \trecall:%f \n'%(total_precision[(i+1)*total_reference-1], total_recall[(i+1)*total_reference-1]))
        f.close()

    @staticmethod
    def calculate_tp(coordinate_pick, coordinate_reference, threshold):
        if len(coordinate_pick)<1 or len(coordinate_reference)<1:
            print("Invalid coordinate parameters in function calculate_tp()!")
        
        # add a symbol to index whether the coordinate is matched with a reference coordinate
        for i in range(len(coordinate_pick)):
            coordinate_pick[i].append(0)

        tp = 0
        average_distance = 0
        print (len(coordinate_reference))

        for i in range(len(coordinate_reference)):
            coordinate_reference[i].append(0)
            coor_x = coordinate_reference[i][0]
            coor_y = coordinate_reference[i][1]
            neighbour = []
            for k in range(len(coordinate_pick)):
                if coordinate_pick[k][4]==0:
                    coor_mx = coordinate_pick[k][0]
                    coor_my = coordinate_pick[k][1]
                    abs_x = math.fabs(coor_mx-coor_x)
                    abs_y = math.fabs(coor_my-coor_y)
                    length = math.sqrt(math.pow(abs_x, 2)+math.pow(abs_y, 2)) 
                    if length < threshold: 
                        same_n = [] 
                        same_n.append(k)
                        same_n.append(length)
                        neighbour.append(same_n)
            if len(neighbour)>=1: 
                if len(neighbour)>1:
                    neighbour = sorted(neighbour, key = itemgetter(1))
                index = neighbour[0][0]
                # change the symbol to 1, means it matchs with a reference coordinate
                coordinate_pick[index][4] = 1
                # add the distance to the list
                coordinate_pick[index].append(neighbour[0][1])
                coordinate_pick[index].append(coor_x)
                coordinate_pick[index].append(coor_y)
                tp = tp + 1 
                average_distance = average_distance+neighbour[0][1]
                coordinate_reference[i][2] = 1
        average_distance = average_distance/tp
        return tp, average_distance




if __name__ == '__main__':
    picker = Picker()
    picker.pick_particle()
    print ("Done.")

