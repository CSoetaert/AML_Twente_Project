import data_extraction as dt
import numpy as np
import dnn_classifier as dnn

def main(): 
	nb_input = 32
	nb_output = 6

	t_set_d = dt.get_data_from_file_list(dt.LIST_TRAINING_FILE_DATA)
	t_set_l = dt.get_data_from_file_list(dt.LIST_TRAINING_FILE_EVENTS)
	t_set_d = t_set_d.reshape((t_set_d.shape[0]//nb_input),nb_input)
	t_set_l = t_set_l.reshape((t_set_l.shape[0]//nb_output),nb_output)

	v_set_d = dt.get_data_from_file_list(dt.LIST_VALIDATION_FILE_DATA)
	v_set_l = dt.get_data_from_file_list(dt.LIST_VALIDATION_FILE_EVENTS)
	v_set_d = v_set_d.reshape((v_set_d.shape[0]//nb_input),nb_input)
	v_set_l = v_set_l.reshape((v_set_l.shape[0]//nb_output),nb_output)


	l_h_u = [32,32,32,32,32,32]
	dnn.dnn_classifier(t_set_d,t_set_l,v_set_d,v_set_l,nb_input,nb_output,l_h_u,1000000,"archi1")
	

if __name__ == "__main__":
	main()