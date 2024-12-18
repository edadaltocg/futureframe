# 💠 Future Frame

Empowering Data Scientists with Foundation Models for Tabular Data

- This Python package allows you to interact with pre-trained foundation models for tabular data.
- Easily fine-tune them on your classification and regression use cases in a single line of code.

## Installation

```bash
pip install futureframe
```

## Quick Start

Use Future Frame to fine-tune a pre-trained foundation model on a classification task.

```python linenums="1"
# Import standard libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import futureframe as ff

# Import data
dataset_name = "https://raw.githubusercontent.com/futureframeai/futureframe/main/tests/data/churn.csv"
target_variable = "Churn"
df = pd.read_csv(dataset_name)

X, y = df.drop(columns=[target_variable]), df[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Fine-tune a pre-trained classifier with Future Frame
model = ff.models.cm2.CM2Classifier()
model.finetune(X_train, y_train)

y_pred = model.predict(X_test)
auc = roc_auc_score(y_test, y_pred)
print(f"AUC: {auc:0.2f}")
```

## Models

| Model Name       | Paper Title                                                                | Paper                                                                               | GitHub                                                    |
| ---------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------- |
| CM2              | Towards Cross-Table Masked Pretraining for Web Data Mining                 | [Ye et al., 2024](https://arxiv.org/abs/2307.04308)                                 | [Link](https://github.com/Chao-Ye/CM2)                    |
| CARTE (soon!)    | CARTE: Pretraining and Transfer for Tabular Learning                       | [Kim et al., 2024](https://arxiv.org/abs/2402.16785)                                | [Link](https://github.com/soda-inria/carte)               |
| TabText (soon!)  | TabText: A Flexible and Contextual Approach to Tabular Data Representation | [Carballo et al., 2023](https://arxiv.org/abs/2206.10381)                           | -                                                         |
| TabPFN (soon!)   | TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second | [Hollmann et al., 2022](https://arxiv.org/abs/2207.01848)                  | [Link](https://github.com/automl/TabPFN)                  |
| TransTab (soon!) | Transtab: Learning Transferable Tabular Transformers Across Tables         | [Wang et al., 2022](https://arxiv.org/abs/2205.09328)                               | [Link](https://github.com/RyanWangZf/transtab)            |

More models will be integrated into the library soon!

More to come!

## Important links

- [Future Frame API Reference](https://github.com/edadaltocg/futureframe-openapi-spec.git)
- [`futureframe` PyPI Page](https://pypi.python.org/pypi/futureframe)
- [`futureframe` GitHub Repository](https://github.com/edadaltocg/futureframe)
<!-- - [`futureframe` Documentation](https://futureframe.ai/docs/) -->

## Contributing

- We are currently under heavy development.
- To report a bug, please write an [issue](https://github.com/edadaltocg/futureframe/issues/new).
