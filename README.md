# Brain Tumor Classification using a Vision Transformer

This repository contains the code for the brain tumor classification using the MRI images. The dataset can be found and downloaded from the following link: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data).

### Project structure

```
braintumorclassification/
│
├── models/
│   └── model_lr_0.01_epoch_19.pth
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
├── .env
├── .gitignore
├── main.ipynb
├── README.md
└── requirements.txt
```

The `models` folder should contain the best performing model on the validation data - saved in the PyTorch format. As the model is too large to be uploaded to the repository - even Git LFS did not work - the model training need to be repeated in order to obtain the model. **Therefore, the evaluation results are only reproducible by running the whole training process.**

Under `src/data` the training and testing data can be found. It is logically split up into the two folders: `train` and `test`. The data is divided into the four classes: `glioma`, `meningioma`, `notumor`, and `pituitary`. The `augmentation.py` file contains the custom class for center cropping the images by a given percentage. The `brain_dataset.py` file contains the custom class for creating a Dataset, the `data_handler.py` file includes the class for preparing the dataset for training and testing.

Under `src/graphics` some images that are being used in the corresponding paper can be found.

The `src/model.py` file contains the customized Vision Transformer model class. The `src/utils.py` file contains the utility functions for setting the seed and logging into W&B.

The `main.ipynb` file is crucial and contains the code for data preprocessing, training and testing the model. All cells and files are well-documented and can be easily followed. The code for creating visualizations is only sparsely commented, as it is not crucial for the understanding of the project.

The `wandb` folder contains the W&B files for tracking the experiments. It is not part of the repository due to its size and personal information that might be included.

In the `.env` file, the W&B project details are stored but need to be adapted if the sweep is executed again. The vars are: `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_IDS`.

### Installation

The python version used is 3.9.13. For reproducibility, create a virtual environment and install the required libraries using the following commands:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

In order to accelerate the training process on Apple silicon such as M1, please manually install the `pytorch` and `torchvision` libraries using the following commands:

```bash
pip install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

Only in that way, one can make use of Metal acceleration on Apple silicon ([Metal Performance Shaders](https://developer.apple.com/metal/pytorch/)).

### W&B Lifecycle Management

The project uses W&B for tracking the experiments. In order to use W&B, please create an account on the [W&B website](https://wandb.ai/). Personal W&B credentials will be asked for during the first run of the `main.ipynb` file. The hyperparameters, metrics, and artifacts are logged into the W&B dashboard during the training process. The `main.ipynb` file contains the code for logging into W&B. The `wandb` folder contains the W&B files for tracking the experiments. An excerpt of the train and val loss curves can be found in the `main.ipynb` file.

Due to the size of the logged artifacts (models), only the best model is included in the repository. The `main.ipynb` is already set up to load the model for further inference and testing.

### Paper

Together with this repository, a corresponding arXiv-style paper has been published. The article includes a detailed explanation of the dataset, the model, and the results. The project doesn't claim to challenge any SOTA results, but rather to provide a concise and comprehensive overview of the process of building a deep learning model for brain tumor classification using a Vision Transformer. The paper can be requested from the author.
