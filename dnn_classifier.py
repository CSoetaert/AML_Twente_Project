import tensorflow as tf
import data_extraction as dt
import numpy as np
import pandas as pd

def dist_prediction(predictions, labels):
	dist = np.zeros(predictions.shape[0])
	for i in range(predictions.shape[0]): 
		dist[i] = np.abs(np.linalg.norm(predictions[i,:]-labels[i,:]))
	return dist


		

def dnn_classifier(training_set_data, training_set_labels, validation_set_data, validation_set_labels, nb_input, nb_output, hidden_units, steps):
	"""fuction containing the structure, training and testing of the dnn classifier
	
	Args:
	    training_set_data (array): contains the data for training
	    training_set_labels (array): contains the labels for training
	    validation_set_data (array): contains the data for validation
	    validation_set_labels (array): contains the labels for validation
	    nb_input (int): nuber of input features
	    nb_classes (int): number of classes
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
		model_dir = "Model")

	training_set_data = training_set_data.reshape((training_set_data.shape[0]//nb_input),nb_input)
	training_set_labels = training_set_labels.reshape((training_set_labels.shape[0]//nb_output),nb_output)


	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x":training_set_data},
		y=training_set_labels,
		num_epochs=None,
		shuffle=True)
	classifier.train(input_fn=train_input_fn,steps=steps)

	validation_set_data = validation_set_data.reshape((validation_set_data.shape[0]//nb_input),nb_input)
	validation_set_labels = validation_set_labels.reshape((validation_set_labels.shape[0]//nb_output),nb_output)

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
	predicted_classes = np.array([p["predictions"] for p in predictions])
	dist = dist_prediction(predicted_classes,validation_set_labels)
	print(np.amax(dist))
	print(np.amin(dist))
	print(np.mean(dist))

	#confusion_matrix = tf.contrib.metrics.confusion_matrix(validation_set_labels,predicted_classes,num_classes=nb_classes)
	#session.run(confusion_matrix)
	#print(confusion_matrix.eval())
	return evaluation#, confusion_matrix