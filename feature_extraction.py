import pyeeg
import data_extraction
import numpy as np
import dnn_classifier as dnn

# bin_power(X, Band, Fs)


def band_power(channel, listOfBands, split_factor):
    """
    uses bin_power(X, Band, Fs) from pyeeg. Ideally it would take a list of channels; wip.
    :param channel: data from one channel only
    :param listOfBands: list of bands to split eeg data into
    :param split_factor: int, how many chunks to split channel into
    :return: list of lists of power in each bin over time chunks
    """

    # split the channel into chucks
    split_channels = np.array_split(channel, split_factor)
    #print(np.shape(split_channels))
    power_list = []

    for chunk in split_channels:
        #print(pyeeg.bin_power(chunk, listOfBands, 500)[0])
        power_list.append(pyeeg.bin_power(chunk, listOfBands, 500)[0])

    #print(np.shape(power_list))

    return np.asarray(power_list)


def summarise_labels(labels, split_factor):
    """

    :param labels: list of labels
    :param split_factor: how many chunks you want to split it into
    :return: list of labels
    """

    # print(np.shape(labels))
    split_labels = np.array_split(labels, split_factor)
    label_summary = []

    for chunk in split_labels:
        #print(np.shape(chunk))
        label_summary.append(int(np.rint(np.mean(chunk, axis=0))))
        #print(int(np.rint(np.mean(chunk, axis=0))))
        #print(np.mean(chunk, axis=0))

    return np.asarray(label_summary)


if __name__ == "__main__":
    data_extraction.save_data(["Data/train/subj1_series2_data.csv"], 'testData')
    data_extraction.save_labels(["Data/train/subj1_series2_events.csv"], 'testLabels')
    data_extraction.save_data(["Data/train/subj1_series3_data.csv"], 'testValData')
    data_extraction.save_labels(["Data/train/subj1_series3_events.csv"], 'testValLabels')


    x = np.load('testData.npy')
    # channel c3
    x = x[:, [12]]
    powerTrain = band_power(x, [0, 4, 8, 12, 16, 31], 1000)

    y = np.load('testLabels.npy')
    y = summarise_labels(y, 1000)

    val = np.load('testValData.npy')
    powerValidation = band_power(val[:, [12]], [0, 4, 8, 12, 16, 31], 1000)

    eventVal = np.load('testValLabels.npy')
    eventVal = summarise_labels(eventVal, 1000)

    dnn.dnn_classifier(powerTrain, y, powerValidation, eventVal, 5, 7, [40,20,30,40,30,20,30,20,35,25,40], 10000)


