##Information gained fro other peoples work on this dataset

The first 2 channels seems to be unuseful because of big artifacts. 

## Result of the models

### All Features Model 

#### Number of hidden units : l_h_u = [40,20,30,40,30,20,30,20,35,25,40]

python3 main.py 
2018-01-25 09:16:06.594919: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
{'average_loss': 0.024901275, 'global_step': 3500000, 'loss': 19.12321}
[[   1842       0       0       0       0       0   37158]
 [      0    2844       0       0       0       0   36156]
 [      0       0    1674       0       0       0   37326]
 [      0       0       0     976       0       0   38024]
 [      0       0       0       0    2624       0   36376]
 [      0       0       0       0       0     772   38228]
 [   2401    2762    2132     935    2505     343 8289274]]

 T = 8300006 F = 234346

 overall accuracy = 0,972540856



python3 onlyc3.py 
2018-01-25 00:21:41.375460: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
{'loss': 20.538303, 'average_loss': 0.026743935, 'global_step': 1000000}
[[      0       0       0       0       0       0   39000]
 [      0       0       0       0       0       0   39000]
 [      0       0       0       0       0       0   39000]
 [      0       0       0       0       0       0   39000]
 [      0       0       0       0       0       0   39000]
 [      0       0       0       0       0       0   39000]
 [      0       0       0       0       0       0 8300352]]


python archi2.py 
2018-01-25 00:20:52.718682: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
{'loss': 20.525331, 'average_loss': 0.026727045, 'global_step': 1000000}
[[      0       0       0       0       0       0   39000]
 [      0       0       0       0       0       0   39000]
 [      0       0       0       0       0       0   39000]
 [      0       0       0       0       0       0   39000]
 [      0       0       0       0       0       0   39000]
 [      0       0       0       0       0       0   39000]
 [      0       0       0       0       0       0 8300352]]

python3 archi1.py 
2018-01-25 09:13:24.261640: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
{'global_step': 2000000, 'average_loss': 0.026697092, 'loss': 20.502329}
[[      0       0       0       0       0       0   39005]
 [      0       0       0       0       0       0   39003]
 [      0       0       1       0       0       0   38999]
 [      0       0       0       0       0       0   39000]
 [      0       0       0       0       0       0   39009]
 [      0       0       0       0       0       0   39031]
 [      0       7      31      16       1       1 8300248]]

empty 
python3 empty.py
2018-01-25 11:33:38.314357: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
{'loss': 21.29169, 'global_step': 1000000, 'average_loss': 0.027724959}
[[      0       0       0       0       0       0   39171]
 [      0       0       0       0       0       0   39339]
 [      0       0       0       0       0       0   39000]
 [      0       0       0       0       0       0   39000]
 [      0       0       0       0       0       0   39171]
 [      0       0       0       0       0       0   39171]
 [      0       0    1408     471       0       0 8297621]]

 
