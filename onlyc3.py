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

	t_set_d = t_set_d[:,12]
	v_set_d = v_set_d[:,12]

	l_h_u = [40,20,30,40,30,20,30,20,35,25,40]
	dnn.dnn_classifier(t_set_d,t_set_l,v_set_d,v_set_l,1,nb_output,l_h_u,500000,"OnlyC3")
	

if __name__ == "__main__":
	main()