# ml_sandbox

> A collection of machine learning projects, standalone scripts, tutorials and
> reference notes — built while exploring applied ML on real-world datasets.

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

Each project is a self-contained Jupyter notebook that walks through a problem
end to end: framing the question, exploring and cleaning the data, engineering
features, modelling, and interpreting the results. The repository doubles as a
sandbox for trying out libraries and techniques (scikit-learn pipelines, PCA,
NLP, PyTorch, Numba JIT).

## Contents

- [Tech stack](#tech-stack)
- [Repository structure](#repository-structure)
- [Applied ML projects](#applied-ml-projects)
- [Python scripts](#python-scripts)
- [PyTorch tutorials](#pytorch-tutorials)
- [Reference notes](#reference-notes)
- [Getting started](#getting-started)
- [License](#license)

## Tech stack

Python · NumPy · pandas · scikit-learn · matplotlib · seaborn · Keras /
TensorFlow · PyTorch · NLTK · Numba

## Repository structure

```
ml_sandbox/
├── notebook-projects/     # End-to-end applied ML projects (Jupyter)
├── py_projects/           # Standalone, runnable Python scripts
├── pytorch_tutorials/     # PyTorch "60-minute blitz" tutorial notebooks
├── reference-notes/       # Course notes and interview-prep notebooks
├── data/                  # Datasets used by the projects
├── requirements.txt       # Python dependencies
└── README.md
```

## Applied ML projects

Notebooks live in [`notebook-projects/`](notebook-projects). GitHub renders the
saved charts and outputs inline, so each one can be read without running it.

| Project | Problem | Techniques | Outcome |
| --- | --- | --- | --- |
| [SMS spam classifier](notebook-projects/spam_classifier_nlp.ipynb) | Classify SMS messages as spam or ham ([dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)) | Text cleaning, tokenisation, bag-of-words / vectorisation; logistic regression, decision tree, and a Keras neural network | >95% test accuracy across all three models ([Kaggle submission](https://www.kaggle.com/philpap/logisticregression-decisiontree-neuralnetwork)) |
| [Credit card churn](notebook-projects/credit_card_churn.ipynb) | Predict which bank customers will churn (BankChurners dataset) | EDA, encoding of nominal features, feature scaling, scikit-learn `Pipeline`s, logistic regression | End-to-end churn classifier with an evaluated accuracy score |
| [Breast cancer — PCA](notebook-projects/breast_cancer_pca.ipynb) | Dimensionality reduction on the scikit-learn breast cancer dataset | Standardisation, Principal Component Analysis | Quantifies variance retained vs. number of components, and reconstruction error |
| [Diabetes — linear regression](notebook-projects/diabetes_linear_regression.ipynb) | Predict disease progression on the diabetes dataset | EDA, supervised linear regression | Fitted regression model with performance metrics |
| [WHOOP sleep EDA](notebook-projects/whoop_sleep_eda.ipynb) | First ML project on personal WHOOP fitness-tracker data | Real-world data cleaning, EDA, linear regression on sleep metrics | Identifies which metrics most influence the WHOOP sleep score |
| [HYROX race simulation](notebook-projects/hyrox_simulation.ipynb) | Estimate finish times for a [HYROX](https://hyrox.com/) fitness race | Monte Carlo simulation; each run leg and station modelled with normal/uniform distributions | Distribution of simulated finish times built from per-station models |

## Python scripts

Runnable scripts in [`py_projects/`](py_projects):

| Script | Description |
| --- | --- |
| [`breast_cancer_pca_example.py`](py_projects/breast_cancer_pca_example.py) | PCA on the breast cancer dataset: shows that retaining all components reconstructs the data almost exactly, and how reconstruction error grows as components are dropped. |
| [`whoop_source_eda.py`](py_projects/whoop_source_eda.py) | Linear-regression model of WHOOP sleep scores, using standardised-coefficient magnitudes as a feature-importance ranking to build a reduced model. |
| [`numba_example.py`](py_projects/numba_example.py) | Demonstrates [Numba](https://numba.pydata.org/) JIT compilation by timing a function on its first (compiled) and subsequent calls. |

```bash
python py_projects/breast_cancer_pca_example.py
python py_projects/numba_example.py
# whoop_source_eda.py needs a personal WHOOP export (not committed):
WHOOP_DATA_DIR=/path/to/whoop_export python py_projects/whoop_source_eda.py
```

## PyTorch tutorials

Worked through [`pytorch_tutorials/`](pytorch_tutorials), following the official
[Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
(run on Google Colab for GPU access):

- **Tensors** — introduction to PyTorch tensors
- **Autograd** — automatic differentiation
- **Neural networks** — building networks with `torch.nn`
- **CIFAR-10** — image classification across 10 classes ([dataset](https://www.cs.toronto.edu/~kriz/cifar.html))
- **Data loading** — working with datasets and dataloaders

## Reference notes

In [`reference-notes/`](reference-notes):

- **`coursera_ml_notes.ipynb`** — key takeaways from Andrew Ng's
  [Coursera Machine Learning course](https://www.coursera.org/learn/machine-learning)
  (gradient descent, classification, overfitting/regularisation)
- **`data_science_interview_questions.ipynb`** — notes on common data science
  and statistics interview topics

## Getting started

```bash
git clone https://github.com/philip-papasavvas/ml_sandbox.git
cd ml_sandbox

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook                 # open any notebook under notebook-projects/
```

The breast cancer and diabetes datasets ship with scikit-learn; the SMS and
BankChurners datasets are included under [`data/`](data). The WHOOP project
uses a personal data export that is not committed to the repository.

## License

Released under the [MIT License](LICENSE).
