import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import time

def calculate_stats(file_name):
    ''' Calculates mean and median of numbers to 1 decimal place '''
    data = np.loadtxt(file_name, delimiter=',')
    mean = np.mean(data)
    median = np.median(data)
    return (np.round(mean, decimals=1), np.round(median, decimals=1))

def mean_datasets(filenames):
    ''' Returns an array with mean of each cell in data files '''
    dataset = np.loadtxt(filenames[0], delimiter=',')
    for filename in filenames[1:]:
        dataset += np.loadtxt(filename, delimiter=',')
    data_mean = dataset/len(filenames)
    return np.round(data_mean, decimals=1)

def load_fits(filename):
    ''' Returns position of brightest pixel on given FITS file '''
    hdulist = fits.open(filename)
    data = hdulist[0].data
    argmax = np.argmax(data)
    max_pos = np.unravel_index(argmax, data.shape)
    return max_pos

def plot_fits(filename):
    ''' Plots image of 2D array of FITS file '''
    plt.imshow(fits.open(filename)[0].data)
    plt.xlabel('x-pixels (RA)')
    plt.ylabel('y-pixels (Dec)')
    plt.colorbar()
    plt.show()

def mean_fits(filenames):
    ''' Returns mean of all pixels in the stack of FITS images '''
    stack = fits.open(filenames[0])[0].data
    if len(filenames) > 1:
        for filename in filenames[1:]:
            stack += fits.open(filename)[0].data
    image_mean = stack/len(filenames)
    return image_mean

def list_stats(input_list):
    ''' Returns tuple of median and mean of input list '''
    input_list.sort()
    mid = len(input_list) // 2
    if len(input_list) % 2 == 0:
        median = (input_list[mid - 1] + input_list[mid]) / 2
    else:
        median = input_list[mid]
    mean = sum(input_list) / len(input_list)
    return(median, mean)

def time_stat(func, size, ntrials):
    ''' Returns the average time for some function to run on a given size of dataset '''
    total = 0
    for _ in range(ntrials):
        data = np.random.rand(size)
        start = time.perf_counter()
        func(data)
        total += time.perf_counter() - start
    return total / ntrials

def median_fits(filenames):
    ''' Returns the median of each pixel over all FITS files, memory and time used for calculation '''
    start = time.perf_counter()
    array = []
    for filename in filenames:
        data = fits.open(filename)[0].data
        array.append(data)
    stack = np.dstack(array)
    median = np.median(stack, axis=2)
    memory = stack.nbytes
    memory /= 1024
    stop = time.perf_counter() - start
    return (median, stop, memory)

def median_bins(values, B):
    ''' Returns mean, standard deviation and the bins from list of values and given B bins '''
    # calculating mean and standard deviations
    mean = np.mean(values)
    std = np.std(values)
    # setting min and max values
    min_val = mean - std
    max_val = mean + std
    # creating an array for bins and setting width
    bins = np.zeros(B)
    ignore_bin = 0
    bin_width = 2 * std / B
    # counting values for each bin
    for value in values:
        if value < min_val:
            ignore_bin += 1
        elif value < max_val:
            bin = int((value - (mean - std)) / bin_width)  # finding out which bin it is in
            bins[bin] += 1

    return (mean, std, bin_width, ignore_bin, bins)
    
def median_approx(values, B):
    ''' Returns approximate median of values '''
    mean, std, bin_width, ignore_bin, bins = median_bins(values, B)
    stop = (len(values) + 1) / 2
    total = ignore_bin
    for i, count in enumerate(bins):
        total += count
        if total >= stop:
            break
    
    median = mean - std + (bin_width * (i + 0.5))
    return median

def running_stats(filenames):
    ''' Returns running mean and standard deviation for a list of FITS files using Welford's method '''
    n = 0
    for filename in filenames:
        hdulist = fits.open(filename)
        data = hdulist[0].data
        if n == 0:
            mean = np.zeros_like(data)
            std = np.zeros_like(data)
        n += 1
        delta = data - mean
        mean += delta / n
        std += delta * (data - mean)
        hdulist.close()
    std /= n - 1
    np.sqrt(std, std)

    if n < 2:
        return mean, None
    else:
        return mean, std

def median_bins_fits(filenames, B):
    ''' Returns mean, standard deviation and the bins from FITS files and given B bins '''
    mean, std = running_stats(filenames)
    dimensions = mean.shape
    # Initialise bins
    ignore_bin = np.zeros(dimensions)
    bins = np.zeros((dimensions[0], dimensions[1], B))
    bin_width = 2 * std / B 

    for filename in filenames:
        hdulist = fits.open(filename)
        data = hdulist[0].data
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                value = data[i, j]
                value_mean = mean[i, j]
                value_std = std[i, j]

                if value < value_mean - value_std:
                    ignore_bin[i, j] += 1
                    
                elif value >= value_mean - value_std and value < value_mean + value_std:
                    bin = int((value - (value_mean - value_std)) / bin_width[i, j])
                    bins[i, j, bin] += 1

    return dimensions, bin_width, mean, std, ignore_bin, bins

def median_approx_fits(filenames, B):
    ''' Returns approximate median of FITS files '''
    dimensions, bin_width, mean, std, ignore_bin, bins = median_bins_fits(filenames, B)
    
    stop = (len(filenames) + 1) / 2
    median = np.zeros(dimensions)   
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):    
            total = ignore_bin[i, j]
            for k, count in enumerate(bins[i, j]):
                total += count
                if total >= stop:
                    break
            median[i, j] = mean[i, j] - std[i, j] + bin_width[i, j] * (k + 0.5)
        
    return median