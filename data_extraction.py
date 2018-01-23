import numpy as np
import timeit
import pandas as pd

LIST_TRAINING_FILE_DATA = []
LIST_TRAINING_FILE_EVENTS = []
LIST_TEST_FILE_DATA = []

for i in range(1,2):
	LIST_TRAINING_FILE_DATA = LIST_TRAINING_FILE_DATA + ["Data/train/subj"+ \
	str(i)+"_series"+str(j)+"_data.csv" for j in range(1,7)]
	LIST_VALIDATION_FILE_DATA = LIST_TRAINING_FILE_DATA + ["Data/train/subj"+ \
	str(i)+"_series"+str(j)+"_data.csv" for j in range(7,9)]
	LIST_TRAINING_FILE_EVENTS = LIST_TRAINING_FILE_EVENTS + ["Data/train/subj" \
	+str(i)+"_series"+str(j)+"_events.csv" for j in range(1,7)]
	LIST_VALIDATION_FILE_EVENTS = LIST_TRAINING_FILE_EVENTS + ["Data/train/subj" \
	+str(i)+"_series"+str(j)+"_events.csv" for j in range(7,9)]
	LIST_TEST_FILE_DATA = LIST_TEST_FILE_DATA + ["Data/test/subj"+str(i)+ \
	"_series"+str(j)+"_data.csv" for j in range(9,11)]



def read_csv_get_values(file_name) : 
	"""Function to read the csv files containing the data and put it
	in a list of list. Each sublist containing the values for one 
	time-step.
	
	Args:
		file_name (TYPE): relative path to the csv file
	
	Returns:
		list: list of the list containing the classes at
		each time step.
	"""
	list_row = pd.read_csv(file_name).as_matrix()
	return list_row

def get_class_from_data(array_row):
	"""Function to get the class from the events file from the data set. 
	
	Args:
		array_row (numpy array): array of the events for each class
	
	Returns:
		numpy array: array containing the class identifier for each time step
	"""
	labels = np.zeros(array_row.shape[0],dtype = int)
	for i in range(array_row.shape[0]):
		labels[i] = 6
		for j in range(array_row.shape[1]):
			if array_row[i,j] == 1.0:
				labels[i] = int(j)
	return labels

def save_data(list_name_file_data):
	"""function to save the datasets in a numpy file
	
	Args:
		list_name_file_data (list): list of the filename from which we
		extract the data
		filename (string): the name of the file in which we save the data
	"""
	list_data = np.array([])
	for i in range(len(list_name_file_data)):
		list_data = np.append(list_data,read_csv_get_values(list_name_file_data[i]))
	#np.save(filename,array_data)
	return list_data

def save_labels(list_name_file_events,filename):
	"""function to save the labels in a numpy file
	
	Args:
		list_name_file_events (list): list of the filename from which we
		extract the labels
		filename (string): the name of the file in which we save the labels
	"""
	list_events = []
	for i in range(len(list_name_file_events)):
		list_events = list_events + read_csv_get_values(list_name_file_events[i])
	array_events = np.array(list_events)
	class_array = get_class_from_data(array_events)
	np.save(filename,class_array)

def get_data_matrix(filename):
	return pd.read_csv(filename).as_matrix()[:,1:].astype(float)

def get_data_from_file_list(filelist):
	data = get_data_matrix(filelist[0])
	for i in range(1,len(filelist)):
		data = np.append(data, get_data_matrix(filelist[i]))
	return data




if __name__ == "__main__":
	#save_data(LIST_TRAINING_FILE_DATA,"Data/training_set_data")
	#save_data(LIST_VALIDATION_FILE_DATA,"Data/validation_set_data")
	#save_labels(LIST_TRAINING_FILE_EVENTS,"Data/training_set_labels")
	#save_labels(LIST_VALIDATION_FILE_EVENTS,"Data/validation_set_labels")
	data = get_data_matrix(LIST_TRAINING_FILE_DATA[0])
	print(data)

