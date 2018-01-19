import pyeeg
import data_extraction as dt
import numpy as np
import dnn_classifier as dnn


def band_power(data, list_of_bands, split_factor):
    """
    uses bin_power(X, Band, Fs) from pyeeg. Ideally it would take a list of channels; wip.
    :param data: list of lists. row = channel, column = frames
    :param list_of_bands: list of bands to split eeg data into
    :param split_factor: int, how many chunks to split channel into
    :return: list of lists of power in each bin over time chunks
    """

    power_list = np.zeros((split_factor, 1))

    np.asarray(power_list)

    for channel in data:
        split_channels = np.array_split(channel, split_factor)

        row_of_powers = []

        for chunk in split_channels:
            row_of_powers.append(pyeeg.bin_power(chunk, list_of_bands, 500)[0])

        power_list = np.append(power_list, row_of_powers, axis=1)

    #print(np.shape(power_list))
    #print(np.shape(power_list[:, 1:]))

    return power_list[:, 1:]


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
        label_summary.append(np.rint(np.mean(chunk, axis=0)))
        #print(int(np.rint(np.mean(chunk, axis=0))))
        #print(np.mean(chunk, axis=0))

    return np.asarray(label_summary)


if __name__ == "__main__":
    #train_data = np.append(dt.get_data_matrix(dt.LIST_TRAINING_FILE_DATA[0]),dt.get_data_matrix(dt.LIST_TRAINING_FILE_DATA[1]))
    #train_labels = np.append(dt.get_data_matrix(dt.LIST_TRAINING_FILE_EVENTS[0]), dt.get_data_matrix(dt.LIST_TRAINING_FILE_EVENTS[1]))
    #val_data = np.append(dt.get_data_matrix(dt.LIST_VALIDATION_FILE_DATA[0]), dt.get_data_matrix(dt.LIST_VALIDATION_FILE_DATA[1]))
    #val_labels = np.append(dt.get_data_matrix(dt.LIST_VALIDATION_FILE_EVENTS[0]), dt.get_data_matrix(dt.LIST_VALIDATION_FILE_EVENTS[1]))

    train_data = dt.get_data_matrix("Data/train/subj1_series1_data.csv")
    train_labels = dt.get_data_matrix("Data/train/subj1_series1_events.csv")
    val_data = dt.get_data_matrix("Data/train/subj1_series2_data.csv")
    val_labels = dt.get_data_matrix("Data/train/subj1_series2_events.csv")

    # print(np.shape(train_data))
    # print(train_labels.shape)

    train_data_selected_columns = []
    val_data_selected_columns = []
    #columns_of_interest = [7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20]
    columns_of_interest = [12, 13, 14]

    for column in columns_of_interest:
        train_data_selected_columns.append(train_data[:, column])
        val_data_selected_columns.append(val_data[:, column])

    powerTrain = band_power(train_data_selected_columns, [0, 4, 8, 12, 16, 31], 1000)
    train_labels = summarise_labels(train_labels, 1000)
    powerValidation = band_power(val_data_selected_columns, [0, 4, 8, 12, 16, 31], 1000)
    val_labels = summarise_labels(val_labels, 1000)

    print(powerTrain.shape)
    print(train_labels.shape)

    dnn.dnn_classifier(powerTrain, train_labels, powerValidation, val_labels, np.shape(powerTrain)[1], 6, [40, 20, 30, 40, 30, 20, 30, 20, 35, 25, 40], 10000)


