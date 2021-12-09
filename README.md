## Who Let the Cats Out?
CSCI1470 Fall 2021, Brendan Ho and Alyssa Loo

# TRAINING THE MODEL
- Have the `labels.npy` and `count_vectors.npy` files downloaded from our GCP bucket.
- Run the `model.py` file with arguments in the command line, in the format 
```
python model.py <Model Type> <Training Type> <Train/Load>
<Model Type>: [CONV/DENSE]
<Training Type>: [WHOLE/SPLIT/FREEZESPLIT]
<Train/Load>: [TRAIN/LOAD]
```

For example
```python model.py DENSE WHOLE TRAIN```

# CALLING THE MODEL FOR PREDICTIONS
- Have the `output_dictionary.pkl` file downloaded from our GCP bucket.
- Open the `demo.py` file. Replace the definitions at the top of the files with the right paths to the requested objects.
- Run the `demo.py` file.

