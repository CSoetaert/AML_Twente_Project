import tensorflow as tf
import data_extraction as dt
import numpy as np

FILE_TRAIN_SUB1_SER1_DATA = "Data/train/subj1_series1_data.csv"
FILE_TRAIN_SUB1_SER1_EVENTS = "Data/train/subj1_series1_events.csv"

FILE_TEST_SUB1_SER2_DATA = "Data/train/subj1_series2_data.csv"
FILE_TEST_SUB1_SER2_EVENTS = "Data/train/subj1_series2_events.csv"

def main(): 
	session = tf.InteractiveSession()
	features = [tf.feature_column.numeric_column("x",shape=[32])]
	classifier = tf.estimator.DNNClassifier(
		feature_columns = features,
		hidden_units = [40,20,30,40,30,20,30,20,35,25,40],
		n_classes = 7,
		model_dir = "Model")

	data_sub1_ser1 = np.array(dt.read_csv_get_values(FILE_TRAIN_SUB1_SER1_DATA))
	events_sub1_ser1 = dt.get_class_from_data(np.array(dt.read_csv_get_values(FILE_TRAIN_SUB1_SER1_EVENTS)))

	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x":data_sub1_ser1},
		y=events_sub1_ser1,
		num_epochs=None,
		shuffle=True)
	classifier.train(input_fn=train_input_fn,steps=4000)

	data_sub1_ser9 = np.array(dt.read_csv_get_values(FILE_TEST_SUB1_SER2_DATA))
	events_sub1_ser9 = np.array(dt.read_csv_get_values(FILE_TEST_SUB1_SER2_EVENTS))
	labels_sub1_ser9 = dt.get_class_from_data(events_sub1_ser9)
	test_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": data_sub1_ser9},
		y=labels_sub1_ser9,
		num_epochs=1,
		shuffle=False)

	evaluation = classifier.evaluate(input_fn=test_input_fn)
	print(evaluation)

	predict_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": data_sub1_ser9},
		num_epochs=1,
		shuffle=False)

	predictions = classifier.predict(input_fn=predict_input_fn)
	predicted_classes = [int(p["classes"][0]) for p in predictions]
	#print(predicted_classes)
	confusion_matrix = tf.contrib.metrics.confusion_matrix(labels_sub1_ser9,predicted_classes,num_classes=7)
	session.run(confusion_matrix)
	print(confusion_matrix.eval())

if __name__ == "__main__":
	main()
