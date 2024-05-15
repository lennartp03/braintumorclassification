# Brain Tumor Classification

This repository contains the code for the brain tumor classification using the MRI images. The dataset can be found and downloaded from the following link: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data).

### Project structure

```
braintumorclassification/
│
├── src/
│   ├── data/
│   │   ├── test/
│   │   │   ├── glioma/
│   │   │   ├── meningioma/
│   │   │   ├── notumor/
│   │   │   └── pituitary/
│   │   ├── train/
│   │   │   ├── glioma/
│   │   │   ├── meningioma/
│   │   │   ├── notumor/
│   │   │   └── pituitary/
│   │   ├── augmentation.py
│   │   ├── brain_dataset.py
│   │   └── data_handler.py
│   ├── graphics/
│   │   ├── ... (images)
│   ├── model.py
│   └── utils.py
│
├── .gitignore
├── main.ipynb
├── requirements.txt
└── README.md
```

Under `src/data` the training and testing data can be found. It is logically split up into the two folders: `train` and `test`. The data is divided into the four classes: `glioma`, `meningioma`, `notumor`, and `pituitary`. The `augmentation.py`file contains the custom class for center cropping the images by a given percentage. The `brain_dataset.py`file contains the custom class for creating a Dataset, the `data_handler.py`file includes the class for preparing the dataset for training and testing.

Under `src/graphics` some images that are being used in the corresponding paper can be found.

The `model.py`file contains the customized Vision Transformer model class. The `utils.py` file contains the utility functions for setting the seed and logging into MLflow.

The `main.ipynb` file is crucial and contains the code for data preprocessing, training and testing the model.

### Installation

To run the code, you need to have the following libraries installed:

- numpy
- avalanche-lib
- matplotlib
- mlflow
- scikit-learn
- torch
- torchvision
- torchinfo
- tqdm
- opencv-python

The python version used is 3.9.13. For reproducibility, create a virtual environment and install the required libraries using the following commands:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Paper

Together with this repository, a corresponding arXiv-style paper has been published. The article includes a detailed explanation of the dataset, the model, and the results. The project doesn't claim to challenge any SOTA results, but rather to provide a concise and comprehensive overview of the process of building a deep learning model for brain tumor classification using a Vision Transformer. The paper can be requested from the author.
