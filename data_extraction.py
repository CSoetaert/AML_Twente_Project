import csv


def read_csv_get_values_eeg(file_name) : 
	"""Function to read the csv files containing the eeg and put it
	in a list of dictionnary. Each dictionnary containing the activity 
	registered at a time step. 
	
	Args:
	    file_name (string): the relative path to the csv file containing
	    the data
	
	Returns:
	    list : list of the dictionnaries containing the activity 
	    registered at each time step.
	"""
	list_fieldnames = ['id', 'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4',
	 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 
	 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 
	 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10']
	with open(file_name, 'rb') as csvfile:
		csv_content = csv.DictReader(csvfile,fieldnames = list_fieldnames)
		first_row = csv_content.next()
		list_row = []
		for row in csv_content:
			for key in row.keys() : 
				if key != 'id' : 
					row[key] = float(row[key])
			list_row.append(row)
	return list_row

def read_csv_get_values_classes(file_name) : 
	"""Function to read the csv files containing the classes and put it
	in a list of dictionnary. Each dictionnary containing the belonging 
	to a class or not at a time step.
	
	Args:
	    file_name (TYPE): Description
	
	Returns:
	    list : list of the dictionnaries containing the classes at 
	    each time step.
	"""
	list_fieldnames = ['id', 'HandStart', 'FirstDigitTouch', 
	'BothStartLoadPhase', 'LiftOff', 'Replace', 'BothReleased']
	with open(file_name, 'rb') as csvfile:
		csv_content = csv.DictReader(csvfile,fieldnames = list_fieldnames)
		first_row = csv_content.next()
		list_row = []
		for row in csv_content:
			for key in row.keys() : 
				if key != 'id' : 
					row[key] = float(row[key])
			list_row.append(row)
	return list_row

data_sub1_ser1 = read_csv_get_values_eeg("Data/train/subj1_series1_data.csv")
print data_sub1_ser1[0]

events_sub1_ser1 = read_csv_get_values_classes("Data/train/subj1_series1_events.csv")
print events_sub1_ser1[0]