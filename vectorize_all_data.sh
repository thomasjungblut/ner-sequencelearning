#!/bin/bash

seq_len=10
emb=data/glove.6B.50d.txt
emb_dim=50

java -jar vectorizer.jar -d ${emb_dim} -e ${emb} -s ${seq_len}
java -jar vectorizer.jar -d ${emb_dim} -e ${emb} -s ${seq_len} -o data_test_a/ -i data/eng.testa.txt -l data/meta.yaml
java -jar vectorizer.jar -d ${emb_dim} -e ${emb} -s ${seq_len} -o data_test_b/ -i data/eng.testb.txt -l data/meta.yaml

