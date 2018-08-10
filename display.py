import numpy as np
import os
from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt
import scipy.misc
from skimage.draw import circle_perimeter

#def plot_circle_in_micrograph(micrograph_2d, reference, coordinate, particle_size, filename, color = 'white'):
def plot_circle_in_micrograph(micrograph_2d, coordinate, particle_size, filename, color = 'white'):
    """plot the particle circle in micrograph image 

    Based on the coordinate of particle, plot circles of the particles in the micrograph.
    And save the ploted image in filename.
 
    Args:
        micrograph_2d: numpy.array,it is a 2D numpy array.
        coordinate: list, it is a 2D list, the shape is (num_particle, 2).
        particle_size: int, the value of the particle size
        filename: the filename of the image to be save.
        color: define the color of the circle

    Raises:
        pass
    """
    display_coordinate = False
    debug = False
    #print len(reference), len(coordinate)
    micrograph_2d = micrograph_2d.reshape(micrograph_2d.shape[0], micrograph_2d.shape[1])
    #print micrograph_2d[0][0:100]
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #plt.axis('off')
    #plt.gray()
    #plt.imshow(micrograph_2d)
    xlen = micrograph_2d.shape[0]
    ylen = micrograph_2d.shape[1]
    radius = particle_size/2
    i = 0 
    #print reference[0]
    array_pick = []
    for l in coordinate:
        x1 = l[0] - radius
        x2 = l[0] + radius
        y1 = l[1] - radius
        y2 = l[1] + radius
        if x1 < 0 or x2 >= xlen or y1 < 0 or y2 >= ylen:
            continue
        #print (left, right, up, down, xlen, ylen)
        patch = np.copy(micrograph_2d[x1:x2, y1:y2])
        array_pick.append(patch)
    array_pick = np.asarray(array_pick)
    if debug:
        print (micrograph_2d.shape)
        print "radius = ", radius
        print coordinate[0]
        print array_pick.shape
        print "pick num = ", len(coordinate)
    pickpath = os.path.join(filename.split('/')[0], 'pick-'+filename.split('/')[1])
    #show_particle(np.asarray(array_pick), pickpath)
    while True: 
        if i >= len(coordinate):
            break
        coordinate_x = coordinate[i][0]
        coordinate_y = coordinate[i][1]
        if display_coordinate:
            print (coordinate_x, coordinate_y)
        if coordinate_x + radius >= xlen or coordinate_y + radius >= ylen or \
            coordinate_x - radius < 0 or coordinate_y - radius < 0:
            i = i + 1
            continue
        rr, cc = circle_perimeter(coordinate_x, coordinate_y, radius)
        micrograph_2d[cc, rr] = micrograph_2d.max()

        #cir1 = Circle(xy = (coordinate_x, coordinate_y), radius = radius, alpha = 0.5, color = color, fill = False)
        #for x in range(-10,10):
        #    for y in range(-10,10):
        #        cir = Circle(xy = (coordinate_x+x, coordinate_y+y), radius = 1, alpha = 0.5, color = color, fill = False)
        #        ax.add_patch(cir)

        #ax.add_patch(cir1)
        # extract the particles
        i = i + 1
    #i = 0
    #while True:
    #    if i >= len(reference):
    #        break
    #    coordinate_x = coordinate[i][0]
    #    coordinate_y = coordinate[i][1]
    #    if coordinate_x + radius >= xlen or coordinate_y + radius >= ylen or \
    #        coordinate_x - radius < 0 or coordinate_y - radius < 0:
    #        i = i + 1
    #        continue
    #    rr, cc = circle_perimeter(coordinate_x, coordinate_y, radius)
    #    try:
    #        micrograph_2d[cc, rr] = 0
    #    except:
    #        print "OH no"
    #    i = i + 1
    scipy.misc.imsave(filename, micrograph_2d)
    #plt.savefig(filename)

def save_image(image_2d, filename):
    scipy.misc.imsave(filename, image_2d)

def show_particle(numpy_array, filename):
    print "hehe", numpy_array.shape
    numpy_array_small = numpy_array[:100, ...]
    numpy_array_small = numpy_array_small.reshape(numpy_array_small.shape[0], numpy_array_small.shape[1], numpy_array_small.shape[2])
    plt.figure(1)
    index = 1
    for i in range(10):
        for j in range(10):
            plt.subplot(10, 10, index)
            plt.gray()
            plt.imshow(numpy_array_small[index-1])
            plt.axis('off')
            index = index + 1
    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.99, bottom=0.01, left=0.01, right=0.99)
    plt.savefig(filename) 


