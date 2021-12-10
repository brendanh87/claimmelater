## Who Let the Cats Out?
CSCI1470 Fall 2021, Brendan Ho and Alyssa Loo

# Training the model
- Have the `labels.npy` and `count_vectors.npy` files downloaded from our GCP bucket. Direct download links: [count-vectors.npy(6.8GB)](https://storage.googleapis.com/claimmelater-trained-weights/count-vectors.npy)|[labels.npy(51.5KB)](https://storage.googleapis.com/claimmelater-trained-weights/labels.npy)
- Run the `model.py` file with arguments in the command line, in the format 
```
python model.py <Model Type> <Training Type> <Train/Load>
<Model Type>: [CONV/DENSE]
<Training Type>: [WHOLE/SPLIT/FREEZESPLIT]
<Train/Load>: [TRAIN/LOAD]
```

For example
```python model.py DENSE WHOLE TRAIN```

# Calling model for predictions
- Prepare two .txt files with the target texts for prediction. For best resuts, ensure there are at least 10 000 characters in both of them.
- Have the `output_dictionary.pkl` file downloaded from our GCP bucket. [output_dictionary.pkl(134.4KB)](https://storage.googleapis.com/claimmelater-trained-weights/output_dictionary.pkl)
- Open the `demo.py` file. Replace the definitions at the top of the files with the right paths to the requested objects.
- Run the `demo.py` file.

