bash get_data.sh

python prepare_input_files.py data/austen 'austen.txt' data/austen_clean
python prepare_input_files.py data/shakespeare/ 'shakespeare.txt' data/shakespeare_clean
python prepare_input_files.py data/scikit-learn '*.py' data/sklearn_clean
python prepare_input_files.py data/scalaz '*.scala' data/scalaz_clean

python split_ebooks.py

python train_test_split.py data/austen_clean/ 0.25
python train_test_split.py data/shakespeare_clean/ 0.25
python train_test_split.py data/sklearn_clean/ 0.25
python train_test_split.py data/scalaz_clean/ 0.25

# keras process
python train_keras.py models/model_keras data/sklearn_clean/ data/scalaz_clean --bidirectional
python apply_tagger_keras.py models/model_keras output/sklearn_or_scala_preds_keras data/sklearn_clean/ data/scalaz_clean
python plot_predictions.py output/sklearn_or_scala_preds_keras output/sklearn_or_scala_preds_keras_html

# pytorch process
python train_pytorch.py models/model_pytorch data/sklearn_clean/ data/scalaz_clean --bidirectional
python apply_tagger_pytorch.py models/model_pytorch output/sklearn_or_scala_preds_pytorch data/sklearn_clean/ data/scalaz_clean --bidirectional
python plot_predictions.py output/sklearn_or_scala_preds_pytorch output/sklearn_or_scala_preds_pytorch_html
