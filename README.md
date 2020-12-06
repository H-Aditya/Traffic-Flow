# Traffic Flow Prediction
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).

## Requirement
- Python 3.8.5 
- Tensorflow-gpu 2.3.1
- Keras 2.4.3
- scikit-learn 0.19
- numpy 1.18.5

## Train the model

**Command to train the model:**

```
	python train.py
```

## Experiment

Data are obtained from the Caltrans Performance Measurement System (PeMS). Data are collected in real-time from individual detectors spanning the freeway system across all major metropolitan areas of the State of California.
	
	GPU: GeFroce GTX 1650ti
	dataset: PeMS 5min-interval traffic flow data
	optimizer: RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
	batch_size: 25 


**Command to make predictions with the saved model:**

```
python predictions.py
```

These are the details for the traffic flow prediction experiment.


| Metrics | MAE | MSE | RMSE | MAPE |  R2  | Explained variance score |
| ------- |:---:| :--:| :--: | :--: | :--: | :----------------------: |
| LSTM | 7.56 | 107.47 | 10.36 | 16.56% | 0.9338 | 0.939 |
| CNN | 7.51 | 105.6 | 10.27| 19.97% | 0.934 | 0.9366|

![evaluate](/results.png)

## Copyright
See [LICENSE](LICENSE) for details.
