import tensorflow as tf
import data_extraction as dt
import numpy as np
import dnn_classifier as dnn

def main(): 
	t_set_d = np.load("Data/training_set_data.npy")
	t_set_l = np.load("Data/training_set_labels.npy")
	v_set_d = np.load("Data/validation_set_data.npy")
	v_set_l = np.load("Data/validation_set_labels.npy")
	l_h_u = [40,20,30,40,30,20,30,20,35,25,40]
	dnn.dnn_classifier(t_set_d,t_set_l,v_set_d,v_set_l,32,7,l_h_u,5000)
	

if __name__ == "__main__":
	main()
