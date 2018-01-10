import csv
import numpy as np

LIST_TRAINING_FILE_DATA = []
LIST_TRAINING_FILE_EVENTS = []
LIST_TEST_FILE_DATA = []

for i in range(1,13):
	LIST_TRAINING_FILE_DATA = LIST_TRAINING_FILE_DATA + ["Data/train/subj"+ \
	str(i)+"_series"+str(j)+"_data.csv" for j in range(1,2)]
	LIST_TRAINING_FILE_EVENTS = LIST_TRAINING_FILE_EVENTS + ["Data/train/subj" \
	+str(i)+"_series"+str(j)+"_events.csv" for j in range(1,9)]
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
	with open(file_name, 'rt') as csvfile:
		csv_content = csv.reader(csvfile)
		first_row = next(csv_content)
		list_row = []
		for row in csv_content:
			for i in range(1,len(row)) : 
				row[i] = float(row[i])
			list_row.append(row[1:])
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

def get_training_set_data(list_name_file_data = LIST_TRAINING_FILE_DATA):
	list_data = []
	for i in range(len(list_name_file_data)):
		print(i)
		list_data = list_data + read_csv_get_values(list_name_file_data[i])
	array_data = np.array(list_data)
	return array_data

def get_training_set_labels(list_name_file_events = LIST_TRAINING_FILE_EVENTS):
	list_events = []
	for i in range(len(list_name_file_data)):
		list_data = list_data + read_csv_get_values(list_name_file_events[i])
	array_data = np.array(list_data)
	class_array = get_class_from_data(array_data)
	return class_array

def get_test_set_data(list_name_file_data = LIST_TEST_FILE_DATA):
	list_data = []
	for i in range(len(list_name_file_data)):
		list_data = list_data + read_csv_get_values(list_name_file_data[i])
	array_data = np.array(list_data)
	return array_data




if __name__ == "__main__":
	# print(LIST_TEST_FILE_DATA)
	# print(LIST_TRAINING_FILE_EVENTS)
	# print(LIST_TRAINING_FILE_DATA)
	# data_sub1_ser1 = read_csv_get_values("Data/train/subj1_series1_data.csv")
	# print(data_sub1_ser1[0])

	# events_sub1_ser1 = read_csv_get_values("Data/train/subj1_series1_events.csv")
	# print(events_sub1_ser1[0])
	training_set = get_training_set_data()
	print(len(training_set))

