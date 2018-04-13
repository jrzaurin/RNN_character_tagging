#!/usr/bin/env bash
bash get_data.sh

python prepare_input_files.py data/austen 'austen.txt' data/austen_clean
python prepare_input_files.py data/shakespeare/ 'shakespeare.txt' data/shakespeare_clean
python prepare_input_files.py data/scikit-learn '*.py' data/sklearn_clean
python prepare_input_files.py data/scalaz/ '*.scala' data/scalaz_clean

python split_ebooks.py

python train_test_split.py data/austen_clean/ 0.25
python train_test_split.py data/shakespeare_clean/ 0.25
python train_test_split.py data/sklearn_clean/ 0.25
python train_test_split.py data/scalaz_clean/ 0.25

python train.py models/model_1 data/sklearn_clean/ data/austen_clean
python apply_tagger.py models/model_1 output/austen_sklearn_pr data/sklearn_clean/ data/austen_clean
python plot_predictions.py output/austen_sklearn_pr output/austen_sklearn_html
