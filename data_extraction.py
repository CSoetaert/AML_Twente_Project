import csv

def read_csv_get_values(file_name) : 
	"""Function to read the csv files containing the data and put it
	in a list of list. Each sublist containing the values for one 
	time-step.
	
	Args:
	    file_name (TYPE): relative path to the csv file
	
	Returns:
	    list : list of the list containing the classes at 
	    each time step.
	"""
	with open(file_name, 'rb') as csvfile:
		csv_content = csv.reader(csvfile)
		first_row = csv_content.next()
		list_row = []
		for row in csv_content:
			for i in range(1,len(row)) : 
				row[i] = float(row[i])
			list_row.append(row)
	return list_row

data_sub1_ser1 = read_csv_get_values("Data/train/subj1_series1_data.csv")
print data_sub1_ser1[0]

events_sub1_ser1 = read_csv_get_values("Data/train/subj1_series1_events.csv")
print events_sub1_ser1[0]