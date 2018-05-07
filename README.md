# Tagging Characters with RNNs

The code in this repo is based on [this post](http://nadbordrozd.github.io/blog/2017/06/03/python-or-scala/) by [Nadbor](http://nadbordrozd.github.io/). We just adapted the code so it works with python 3, implement the process using `pytorch`, and use this project to illustrate the differences between `keras` and `pytorch`

## The goal

Our aim here is the following: given some text that comes from multiple sources (e.g two different programming languages or two different writers), can we train a RNN to find which characters belong to each of the sources?

For example, the text below is comprised by bits of `python` and `scala` code stitched together. Can we find out where the `python` ends/begins and the `scala` begins/ends?

```
from __future__ import division
from functools import partial
import warnings

import numpy as np
from scipy import linalg
from scipy.sparse import issparse, csr_matr F[A])(implicit val F: MonadPlus[F]) extends Ops[F[A]] {
////
impoix

from . import check_random_state
from .fixrt Leibniz.===

def filter(f: A => Boolean): F[A] =
F.filter(self)(f)

def withFilter(f: A => Boolean): F[A] =
filter(f)

final def uniteU[T](implicit T: Unapply[Foldable, Aes import np_version
from ._logistic_sigmoid import _log_logistic_sigmoid
from ..extern]): F[T.A] =
F.uniteU(self)(T)

def unite[T[_], B](implicit ev: A === T[B], T: Foldable[T]): F[B] = {
val ftb: F[T[B]] = ev.subst(seals.six.moves import xrange
from .sparsefuncs_fast import csr_row_norms
from .validation import check_array
from ..exceptions import NonBLASDotWarning
```

## The code: keras vs pytorch

Just so is all clear since the beginning, is not that we have a favourite frame. We like both, `keras` and `pytorch`. Our aim here is to illustrate their differences using a relatively simple model with a non-trivial data-preparation process. Having said that, here we go.

There is a `run_all.sh` file that you could run by

```
bash run_all.sh
```

However, I would recommend to use `run_all.sh` to follow the flow first and run script by script, or perhaps better, have a look to:

```
full_process_keras.ipynb
full_process_pytorch.ipynb
```

The full process is detailed in those two notebooks. I would recommend to go first through `full_process_keras.ipynb` and then `full_process_pytorch.ipynb`.


Nonetheless, the *"flow of the code"* is:

1. Get the data. We will download 2 sets of files. One to compare `python` and `scala` sequences of text and another to compare Jane Austen's and Shakespeare's  books

	`bash get_data.sh`

2. Prepare input files:

	```
	python prepare_input_files.py data/austen 'austen.txt' data/austen_clean
	python prepare_input_files.py data/shakespeare/ 'shakespeare.txt' data/shakespeare_clean
	python prepare_input_files.py data/scikit-learn '*.py' data/sklearn_clean
	python prepare_input_files.py data/scalaz '*.scala' data/scalaz_clean
	```

3. I have included a little script to split the books in a way that each partition "makes sense", meaning contains enough text and the text corresponds to chapters, episodes, etc:

	`python split_ebooks.py`

4. train/test split

	```
	python train_test_split.py data/austen_clean/ 0.25
	python train_test_split.py data/shakespeare_clean/ 0.25
	python train_test_split.py data/sklearn_clean/ 0.25
	python train_test_split.py data/scalaz_clean/ 0.25
	```

5. Train the network, with `keras` or `pytorch`, to distinguish between `python` or `scala` code (just change the corresponding directories to compare Austen and Shakespeare)

	```
	python train_keras.py models/model_keras data/sklearn_clean/ data/scalaz_clean --bidirectional
	python train_pytorch.py models/model_pytorch data/sklearn_clean/ data/scalaz_clean --bidirectional
	```

6. Tag the characters as being `python` or `scala` code:

	```
	python apply_tagger_keras.py models/model_keras output/sklearn_or_scala_preds_keras data/sklearn_clean/ data/scalaz_clean
	python apply_tagger_pytorch.py models/model_pytorch output/sklearn_or_scala_preds_pytorch data/sklearn_clean/ data/scalaz_clean
	```

7. And finally plot the output to pretty html files:

	```
	python plot_predictions.py output/sklearn_or_scala_preds_keras output/sklearn_or_scala_preds_keras_html
	python plot_predictions.py output/sklearn_or_scala_preds_pytorch output/sklearn_or_scala_preds_pytorch_html
	```

Again, I **recommend** having a look to the notebooks.

Details on why this is "cool" or can be useful can be found in the already mentioned [post](http://nadbordrozd.github.io/blog/2017/06/03/python-or-scala/) by Nadbor.

Thanks and credit to Nadbor for having the idea and trigger a fun project.

