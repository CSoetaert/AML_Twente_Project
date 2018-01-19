import tensorflow as tf
import data_extraction as dt
import numpy as np
import dnn_classifier as dnn

def main(): 
	t_set_d = np.append(dt.get_data_matrix(dt.LIST_TRAINING_FILE_DATA[0]),dt.get_data_matrix(dt.LIST_TRAINING_FILE_DATA[1]))
	t_set_l = np.append(dt.get_data_matrix(dt.LIST_TRAINING_FILE_EVENTS[0]),dt.get_data_matrix(dt.LIST_TRAINING_FILE_EVENTS[1]))
	v_set_d = np.append(dt.get_data_matrix(dt.LIST_VALIDATION_FILE_DATA[0]),dt.get_data_matrix(dt.LIST_VALIDATION_FILE_DATA[1]))
	v_set_l = np.append(dt.get_data_matrix(dt.LIST_VALIDATION_FILE_EVENTS[0]),dt.get_data_matrix(dt.LIST_VALIDATION_FILE_EVENTS[1]))
	l_h_u = [40,20,30,40,30,20,30,20,35,25,40]
	dnn.dnn_classifier(t_set_d,t_set_l,v_set_d,v_set_l,32,6,l_h_u,10000)
	

if __name__ == "__main__":
	main()
