import random
import numpy as np
import torch
import mlflow

def set_seeds(seed):
    '''
    Set seeds for reproducibility.

    Args:
    -----
    seed: int
        Custom seed value
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_to_mlflow(model, train_acc, train_loss, val_acc, val_loss, ep, lr, batch_size):
    '''
    Logging training parameters, metrics and model to MLflow.

    Args:
    -----
    model: torch.nn.Module
        Model to log
    train_acc: float
        Training accuracy
    train_loss: float
        Training loss
    val_acc: float
        Validation accuracy
    val_loss: float
        Validation loss
    ep: int
        Epoch number
    lr: float
        Learning rate
    batch_size: int
        Batch size
    '''

    mlflow.log_params({'learning_rate': lr,
                       'batch_size': batch_size})

    mlflow.log_metrics({'train_accuracy': train_acc,
                        'train_loss': train_loss,
                        'val_accuracy': val_acc,
                        'val_loss': val_loss
                    }, step=ep)
    
    mlflow.pytorch.log_model(
        model,
        'brain-tumor-vit-classifier',
        registered_model_name=f'brain-tumor-model-{ep}-{lr}',
    )