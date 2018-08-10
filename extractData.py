from dataLoader import DataLoader

import os,sys
from optparse import OptionParser

def extractData():
    parser = OptionParser()
    parser.add_option("--inputDir", dest="inputDir", help="Input directory", metavar="DIRECTORY")
    parser.add_option("--mrc_number", dest="mrc_number", help="Number of mrc files to be trained.", metavar="VALUE", default=-1)
    parser.add_option("--coordinate_symbol", dest="coordinate_symbol", help="The symbol of the coordinate file, like '_manualPick'", metavar="STRING")
    parser.add_option("--particle_size", dest="particle_size", help="the size of the particle.", metavar="VALUE", default=-1)
    parser.add_option("--save_dir", dest="save_dir", help="save the training samples to this directory", metavar="DIRECTORY", default="../trained_model")
    parser.add_option("--save_file", dest="save_file", help="save the training samples to file", metavar="FILE")
    parser.add_option("--produce_negative", dest="produce_negative", help="whether to produce negative samples", metavar="BOOL", default=True)

    parser.add_option("--class_number", dest="class_number", help="ClassNumber", metavar="STRING", default='0')
    (opt, args) = parser.parse_args()

    inputDir = opt.inputDir
    particle_size = int(opt.particle_size)
    coordinate_symbol = opt.coordinate_symbol
    mrc_number = int(opt.mrc_number)
    output_dir = opt.save_dir
    #output_filename = opt.save_file
    produce_negative = opt.produce_negative
    class_number = int(opt.class_number)
    print "produce_negative =", produce_negative
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    print 'mkdir!'

    if particle_size == -1:
        print("particle size should be a positive value!")
        return 

    #output_filename = os.path.join(output_dir, output_filename)
    print 'begin extracting!'
    if class_number == 0:
        output_filename = os.path.join(output_dir, opt.save_file)
        print ("saved_file >>> ", output_filename)
        sys.stdout.flush()
        DataLoader.extractData(inputDir, particle_size, coordinate_symbol, mrc_number, output_filename, produce_negative=produce_negative)
    else:
        #for i in range(1, class_number+1):
        #    output_filename = "class%d.pickle" % i
        #    output_filename = os.path.join(output_dir, output_filename)
        #    print (coordinate_symbol+str(i), " >>> ", output_filename)
        #    sys.stdout.flush()
        #    DataLoader.extractData(inputDir, particle_size, coordinate_symbol+str(i), mrc_number, output_filename, produce_negative=produce_negative)
        DataLoader.extractData(inputDir, particle_size, coordinate_symbol, mrc_number, output_dir, class_number, produce_negative=produce_negative)

def main(argv=None):
    extractData()
    print ("Extract Done.")

if __name__ == '__main__':
    main()
