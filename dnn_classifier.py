import tensorflow as tf
import data_extraction as dt
import numpy as np


def dnn_classifier(training_set_data, training_set_labels, validation_set_data, validation_set_labels, nb_input, nb_classes, hidden_units, steps):
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
	classifier = tf.estimator.DNNClassifier(
		feature_columns = features,
		hidden_units = hidden_units,
		n_classes = nb_classes,
		model_dir = "Model")

	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x":training_set_data},
		y=training_set_labels,
		num_epochs=None,
		shuffle=True)
	classifier.train(input_fn=train_input_fn,steps=steps)

	test_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": validation_set_data},
		y=validation_set_labels,
		num_epochs=1,
		shuffle=False)

	evaluation = classifier.evaluate(input_fn=test_input_fn)
	print(evaluation)

	predict_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": validation_set_data},
		num_epochs=1,
		shuffle=False)

	predictions = classifier.predict(input_fn=predict_input_fn)
	predicted_classes = [int(p["classes"][0]) for p in predictions]
	#print(predicted_classes)
	confusion_matrix = tf.contrib.metrics.confusion_matrix(validation_set_labels,predicted_classes,num_classes=nb_classes)
	session.run(confusion_matrix)
	print(confusion_matrix.eval())
	return evaluation, confusion_matrix