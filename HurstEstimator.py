import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

def func(x, h, c):
    #This is just the function for fitting to find the slope, the hurst exponent
    yData = h*x +c
    return yData

def random_data_maker(npoints):
    #This gives us something to check our hurst estimator against, we expect the Hurst exponent to  be 0.5 for truly random data
    data = np.random.randn(npoints)
    return data

def random__cumsum_data_maker(npoints):
    #Here we just generate a cumulative sum of random data
    data = np.cumsum(np.random.randn(npoints))
    return data


def hurstEstimator(data, maximumtimespan, cumulative, graph):

    """
    :param data: array
                This is the data of which we are estimating the hurst exponent. This should be an array.
    :param maximumtimespan: int
                    This is the divider choosing how to chop up the data. The larger this is the smaller the window.
    :param cumulative: bool
                    If true then the hurstEstimator will decompose the data into single changes between time steps.
    :param graph: bool
                If true this will produce graphs of the data and the least squares fit to find the Hurst exponent.
    :return: float
                The Hurst exponent 0<H<1. If H < 0.5 we have anti-persistence, H = 0.5 truly a random time-series, and H>0.5 we have persistence.
    """

    if cumulative == True:
        data = data[1:len(data)] - data[0:len(data) - 1]
    else:
        data = data

    segmentLengths = np.arange(10, np.floor(len(data) / (maximumtimespan)))
    expectedRescaledRange = np.zeros(len(segmentLengths))

    i = 0
    for j in segmentLengths:
        rescaledRange = np.zeros(len(data) + 1 - int(j))
        # i, = np.where(segmentLengths == j)
        for t in range(len(data) + 1 - int(j)):
            start = t
            end = t + int(j)
            recentedData = data[start:end] - np.mean(data[start:end])
            Z = np.cumsum(recentedData)
            deviateSeriesRange = np.ptp(Z)
            standardDeviation = np.std(data[start:end])
            rescaledRange[t] = deviateSeriesRange / standardDeviation
        expectedRescaledRange[i] = np.mean(rescaledRange)
        i += 1

    xdata = np.log2(np.flip(segmentLengths))
    ydata = np.log2(np.flip(expectedRescaledRange))

    popt, pcov = curve_fit(func, xdata, ydata)
    perr = np.sqrt(np.diag(pcov))
    # print("The estimated Hurst Exponent is %2f"%  popt[0])
    # print("The estimated one standard deviation error is %2f" %perr[0])

    if graph == True:
        plt.figure(figsize=(10, 8))
        plt.subplot(212)
        plt.scatter(xdata, ydata, label='Estimated Rescaled Range Values ')
        plt.plot(xdata, func(xdata, *popt), 'r--', label='Least Squares Regression')
        plt.xlabel('log(n)')
        plt.ylabel('log(R/S)')
        plt.title('Rescaled Ranges')
        plt.legend()

        plt.subplot(221)
        plt.plot(data)
        plt.title('Data')
        plt.xlabel('t')
        plt.ylabel('$X$')

        plt.subplot(222)
        plt.plot(np.cumsum(data))
        plt.xlabel('t')
        plt.ylabel('Cumulative Sum of X')
        plt.title('Cumulative Sum of Data')

    return popt[0]