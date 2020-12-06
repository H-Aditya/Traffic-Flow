# Traffic Flow Prediction
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).

## Requirement
- Python 3.8.5 
- Tensorflow-gpu 2.3.1
- Keras 2.4.3
- scikit-learn 0.19
- numpy 1.18.5

## Train the model

**Run command below to train the model:**

```
python train.py --model model_name
```

You can choose "lstm", "gru" or "saes" as arguments. The ```.h5``` weight file was saved at model folder.


## Experiment

Data are obtained from the Caltrans Performance Measurement System (PeMS). Data are collected in real-time from individual detectors spanning the freeway system across all major metropolitan areas of the State of California.
	
	device: Tesla K80
	dataset: PeMS 5min-interval traffic flow data
	optimizer: RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
	batch_szie: 256 


**Run command below to run the program:**

```
python main.py
```

These are the details for the traffic flow prediction experiment.


| Metrics | MAE | MSE | RMSE | MAPE |  R2  | Explained variance score |
| ------- |:---:| :--:| :--: | :--: | :--: | :----------------------: |
| LSTM | 7.56 | 107.47 | 10.36 | 17.93% | 0.9338 | 0.9390 |
| CNN | 7.51 | 105.63 | 10.27| 19.97 | 0.9349 | 0.9366|

![evaluate](/results.png)

## Reference

	@article{SAEs,  
	  title={Traffic Flow Prediction With Big Data: A Deep Learning Approach},  
	  author={Y Lv, Y Duan, W Kang, Z Li, FY Wang},
	  journal={IEEE Transactions on Intelligent Transportation Systems, 2015, 16(2):865-873},
	  year={2015}
	}
	
	@article{RNN,  
	  title={Using LSTM and GRU neural network methods for traffic flow prediction},  
	  author={R Fu, Z Zhang, L Li},
	  journal={Chinese Association of Automation, 2017:324-328},
	  year={2017}
	}


## Copyright
See [LICENSE](LICENSE) for details.
