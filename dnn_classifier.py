import tensorflow as tf
import data_extraction as dt
import numpy as np
import pandas as pd

def dist_prediction(predictions, labels):
	dist = np.zeros(predictions.shape[0])
	for i in range(predictions.shape[0]): 
		dist[i] = np.abs(np.linalg.norm(predictions[i,:]-labels[i,:]))
	return dist

def confusion_matrix(prediction, labels, nb_states = 7): 
	confusion_matrix = np.zeros((nb_states,nb_states))
	for i in range(prediction.shape[0]):
		for j in range(prediction.shape[1]):
			if prediction[i,j] == labels[i,j]:
				if prediction[i,j] !=0 : 
					confusion_matrix[j,j] +=1
				else : 
					confusion_matrix[nb_states-1,nb_states-1] +=1
			elif prediction[i,j]<labels[i,j]:
				confusion_matrix[j,nb_states-1] +=1
			elif prediction[i,j]>labels[i,j]: 
				confusion_matrix[nb_states-1,j] +=1
	return confusion_matrix.astype(int)


def dnn_classifier(training_set_data, training_set_labels, validation_set_data, validation_set_labels, nb_input, nb_output, hidden_units, steps, model_dir = "Model"):
	"""fuction containing the structure, training and testing of the dnn classifier
	
	Args:
	    training_set_data (array): contains the data for training
	    training_set_labels (array): contains the labels for training
	    validation_set_data (array): contains the data for validation
	    validation_set_labels (array): contains the labels for validation
	    nb_input (int): nuber of input features
	    nb_output (int): number of output unit
	    hidden_units (list): list of the number of neurns for each hidden layer
	    steps (int): number of step of training
	
	Returns:
	    TYPE: Description
	"""
	session = tf.InteractiveSession()
	features = [tf.feature_column.numeric_column("x",shape=[nb_input])]
	classifier = tf.estimator.DNNRegressor(
		feature_columns = features,
		hidden_units = hidden_units,
		label_dimension = nb_output,
		model_dir = model_dir)

	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x":training_set_data},
		y=training_set_labels,
		num_epochs=None,
		shuffle=True)
	classifier.train(input_fn=train_input_fn,steps=steps)


	test_input_fn = tf.estimator.inputs.numpy_input_fn(
		x= {"x":validation_set_data},
		y= validation_set_labels,
		num_epochs=1,
		shuffle=False)

	evaluation = classifier.evaluate(input_fn=test_input_fn)
	print(evaluation)

	predict_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": validation_set_data},
		num_epochs=1,
		shuffle=False)

	predictions = classifier.predict(input_fn=predict_input_fn)

	predicted_classes = np.array([np.around(p["predictions"],0) for p in predictions])
	#print(predicted_classes)
	print(confusion_matrix(predicted_classes,validation_set_labels))
	#dist = dist_prediction(predicted_classes,validation_set_labels)
	#confusion_matrix = tf.contrib.metrics.confusion_matrix(validation_set_labels,predicted_classes,num_classes=nb_classes)
	#session.run(confusion_matrix)
	#print(confusion_matrix.eval())
	return evaluation#, confusion_matrix