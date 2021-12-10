## Who Let the Cats Out?
CSCI1470 Fall 2021, Brendan Ho and Alyssa Loo

###[Final Writeup](https://docs.google.com/document/d/1SldKaEAevO9QALDZy1KmQJWis0di8VJpq5rGMoytfpM/edit?usp=sharing) | [Poster] (https://docs.google.com/presentation/d/1NEXjwr2Q5dsh8_qjlTJOBcEelxZyKfjKr29CO-FsW24/edit?usp=sharing)
# Training the model
- Have the `labels.npy` and `count_vectors.npy` files downloaded from our GCP bucket. 
- Direct download links: [count-vectors.npy (6.8GB) ](https://storage.googleapis.com/claimmelater-trained-weights/count-vectors.npy)| [labels.npy (51.5KB)](https://storage.googleapis.com/claimmelater-trained-weights/labels.npy)
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
- Prepare two .txt files with the target texts for prediction. For best resuts, ensure there are at least 10 000 characters in both of them. There are some sample .txt files in the `hw7/demo_data` folder
- Have the `output_dictionary.pkl` file downloaded from our GCP bucket.
  - Direct download link: [output_dictionary.pkl (134.4KB)](https://storage.googleapis.com/claimmelater-trained-weights/output_dictionary.pkl)
- Have a saved set of weights. You can train the model yourself (above), or download these pre-trained set of weights (whole-trained, all dense layers, addition in resblock):
  - Direct download link (all three files must be in the same directory for Tensorflow to load saved weights): 
    - [checkpoint (95B)](https://storage.googleapis.com/claimmelater-trained-weights/whole_model_weights_addition/checkpoint)
    - [.data (108.9MB)](https://storage.googleapis.com/claimmelater-trained-weights/whole_model_weights_addition/whole_model_weights.data-00000-of-00001)
    - [.index (19KB)](https://storage.googleapis.com/claimmelater-trained-weights/whole_model_weights_addition/whole_model_weights.index)
- Open the `demo.py` file. Replace the definitions at the top of the files with the right paths to the requested objects.
  - For the path to the weights, it should be formatted as `directorypath/weightname`, where weightname is the prefix before `.index` and `.data[...]`
- Run the `demo.py` file.

