import os
import numpy as np
import matplotlib.pyplot as plt


def plot_images(batch, plot_dir, num_examples):
    '''
    Plots images from a batch and saves to a directory
    
    Arguments:
        batch - tensorflow dataset batch,   contains images and labels
        plot_dir - str,                     name of directory in which to save images
        num_examples - int,                 number of examples to plot
    
    Returns:
        nothing, but saves plots to dir
    '''
    images, labels = batch
    images = images.numpy()
    labels = labels.numpy()

    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    for i in range(num_examples):
        image = images[i]
        imshape = image.shape
    
        if len(imshape)==2:
            imshape = imshape 
        elif len(imshape)==3 and imshape[-1] == 3:
            imshape = image.shape
        elif len(imshape)==3 and imshape[-1] == 1:
            imshape = image.shape[:-1]
        else:
            raise ValueError('cannot plot image of shape'.format(str(imshape)))
        
        image = image.reshape(imshape)

        label = labels[i]
        plt.imshow(image)
        plt.savefig(os.path.join(plot_dir,
                                'image_{}_{}.png'.format(str(i), str(label))))