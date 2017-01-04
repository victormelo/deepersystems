## Your prediction on the test data

It's on the file `test_evaluation.csv`

## All the code needed to train and run your network to produce that prediction from scratch, along with instructions on how to run the code

All the code is on the file `model.py` and the dependencies to run the code is on the file `requirements.txt`. The files `train_100k.csv`, `train_100k.truth.csv`, `test_100k.csv` must be unzipped on the folder `data`

To run the code simply do:
`python model.py`

## A short description of the approach you took and how you arrived at the solution you did

My Neural Net is basically a three layer MLP with (100, 50, 20) neurons. The activation function used in the units is the ReLU. The models are trained with 250 epochs.

I trained two models.
The first model was trained with 66% of the data and tested with 34%. I did this so I could get an idea how the model would perform on a "real scenario", by that I mean evaluated with data never seen. 
The results I got with this experiment were:
```
Slope mse: 0.42872882576
Intercept mae: 5.56380679529
```

The second model was trained using all the training data. The predictions are on `train_pred.csv` and results reported below. Finally, I also tested the model with the test data. The predictions are on the file `test_evaluation.` 

- With the network trained on the training set and evaluated on the same set (the same experiment described on the problem README) I had the following results:
```
Slope mse: 0.150009972589
Intercept mae: 4.01338658376
```
