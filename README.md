This is a repo about deep sequence learning in the named entity recognition domain using Keras and Glove. 

Run it!
-------

This task works on the Conll2003 dataset found and described here:
http://www.cnts.ua.ac.be/conll2003/ner/

Get glove embeddings from:
https://nlp.stanford.edu/data/glove.6B.zip

For this training we're using the 50 dimensional embeddings, unzip and copy them into the data directory.
Now transform the conll dataset into a sequence vector representation using the Java program in the repo:

> java -jar vectorizer.jar

You can add -h as an argument for some more options with regards to I/O and sequence lengths.

To vectorize all train and test files consistently, you can also use the bash script:

> ./vectorize_all_data.sh

Then on top of the generated data, you can run the keras model training, which does a 5-fold CV:

> python3 train.py


The model
---------

The model is a Bi-LSTMs over two convolutional layers that is predicting time distributed sequences.
We create fixed length (in this case 10) sequences of words, not caring about alignments or annotations.
Spending more time to chunk into appropriate sized sequences should give even better results.
As features we rely on the glove embeddings, one hot encoded POS-tags retrieved directly from the training set, 
3-char ngrams and some basic word shape features. 

The model summary looks like this:
```
timesteps: 10, input dim: 773, num output labels: 8
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv1d_1 (Conv1D)            (None, 10, 512)           396288
_________________________________________________________________
batch_normalization_1 (Batch (None, 10, 512)           2048
_________________________________________________________________
p_re_lu_1 (PReLU)            (None, 10, 512)           5120
_________________________________________________________________
dropout_1 (Dropout)          (None, 10, 512)           0
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 10, 512)           262656
_________________________________________________________________
batch_normalization_2 (Batch (None, 10, 512)           2048
_________________________________________________________________
p_re_lu_2 (PReLU)            (None, 10, 512)           5120
_________________________________________________________________
dropout_2 (Dropout)          (None, 10, 512)           0
_________________________________________________________________
bidirectional_1 (Bidirection (None, 10, 300)           795600
_________________________________________________________________
batch_normalization_3 (Batch (None, 10, 300)           1200
_________________________________________________________________
p_re_lu_3 (PReLU)            (None, 10, 300)           3000
_________________________________________________________________
dropout_3 (Dropout)          (None, 10, 300)           0
_________________________________________________________________
time_distributed_1 (TimeDist (None, 10, 8)             2408
=================================================================
Total params: 1,475,488
Trainable params: 1,472,840
Non-trainable params: 2,648
_________________________________________________________________
```

Test set performance
--------------------

Conll2003 ships with two test files (A and B). You can generate the consistent vectorized forms using this bash:

> ./vectorize_all_data.sh

and then run the prediction and evaluation with:

> python3 predict_test.py

The test takes each model from the CV folds and averages their prediction result. Below result was an averaged 
ensemble over 5 models with the above model description.     


For test set A we get:

``` 
 processed 51570 tokens with 6178 phrases; found: 6219 phrases; correct: 5758.
accuracy:  98.85%; precision:  92.59%; recall:  93.20%; FB1:  92.89
              LOC: precision:  94.72%; recall:  95.90%; FB1:  95.30  1875
             MISC: precision:  88.42%; recall:  85.70%; FB1:  87.04  915
              ORG: precision:  89.22%; recall:  89.54%; FB1:  89.38  1429
              PER: precision:  94.90%; recall:  96.94%; FB1:  95.91  2000

                  O   I-ORG  I-MISC   I-PER   I-LOC   B-LOC  B-MISC   B-ORG
          O 42869.0    38.0    31.0    15.0    15.0     0.0     0.0     0.0
      I-ORG    59.0  1917.0    31.0    43.0    41.0     0.0     0.0     0.0
     I-MISC    84.0    49.0  1096.0    17.0    17.0     0.0     1.0     0.0
      I-PER    30.0     4.0     7.0  3085.0    23.0     0.0     0.0     0.0
      I-LOC    18.0    42.0    10.0    13.0  2011.0     0.0     0.0     0.0
      B-LOC     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0
     B-MISC     0.0     0.0     3.0     0.0     0.0     0.0     1.0     0.0
      B-ORG     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0
```

and for test set B we get:

``` 
 processed 46660 tokens with 5857 phrases; found: 5947 phrases; correct: 5237.
accuracy:  97.79%; precision:  88.06%; recall:  89.41%; FB1:  88.73
              LOC: precision:  89.08%; recall:  92.57%; FB1:  90.79  1749
             MISC: precision:  80.08%; recall:  78.10%; FB1:  79.08  708
              ORG: precision:  83.55%; recall:  86.30%; FB1:  84.90  1787
              PER: precision:  95.07%; recall:  94.24%; FB1:  94.65  1703

                  O   I-ORG  I-MISC   I-PER   I-LOC   B-LOC  B-MISC   B-ORG
          O 38220.0   153.0   118.0    28.0    31.0     0.0     0.0     0.0
      I-ORG    79.0  2221.0    49.0    31.0   111.0     0.0     0.0     0.0
     I-MISC    69.0    66.0   733.0    12.0    29.0     0.0     0.0     0.0
      I-PER    19.0    47.0     4.0  2675.0    27.0     0.0     0.0     0.0
      I-LOC    17.0    84.0    25.0    10.0  1782.0     0.0     0.0     0.0
      B-LOC     2.0     3.0     0.0     0.0     1.0     0.0     0.0     0.0
     B-MISC     1.0     0.0     6.0     0.0     2.0     0.0     0.0     0.0
      B-ORG     0.0     5.0     0.0     0.0     0.0     0.0     0.0     0.0
```

Vectorizer Compilation
----------------------

The vectorizer itself can be compiled with maven using:

> mvn clean install package

this generates a fatjar (vectorizer-0.0.1-jar-with-dependencies.jar) to execute the vectorizer as above and a jar with just the code itself.

License
-------

Since I am Apache committer, I consider everything inside of this repository 
licensed by Apache 2.0 license, although I haven't put the usual header into the source files.

If something is not licensed via Apache 2.0, there is a reference or an additional licence header included in the specific source file.
