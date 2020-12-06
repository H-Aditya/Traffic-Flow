# Traffic Flow Prediction
Traffic Flow Prediction with Neural Networks(LSTM, CNN).

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
	
	GPU: GeForce GTX 1650Ti
	dataset: PeMS 5min-interval traffic flow data
	optimizer: RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
	batch_szie: 25 


**Command to predict with saved models:**

```
python predictions.py
```

These are the details for the traffic flow prediction experiment.


| Metrics | MAE | MSE | RMSE | MAPE |  R2  | Explained variance score |
| ------- |:---:| :--:| :--: | :--: | :--: | :----------------------: |
| LSTM | 7.56 | 107.47 | 10.36 | 17.93% | 0.9338 | 0.9390 |
| CNN | 7.51 | 105.63 | 10.27| 19.97 | 0.9349 | 0.9366|

![evaluate](/results.png)

## Copyright
See [LICENSE](LICENSE) for details.
