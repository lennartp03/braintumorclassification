import timm
import torch.nn as nn

class VisionTransformerWithCustomHead(nn.Module):
    '''
    A custom Vision Transformer model with a modified MLP head for classification.

    Attributes:
    -----------
    vit_model: timm.models.vision_transformer.VisionTransformer
        Pretrained Vision Transformer model
    features: torch.nn.Sequential
        Sequential container for the feature extractor part of the model
    ln1: torch.nn.Linear
        First linear layer in the custom head
    relu: torch.nn.ReLU
        ReLU activation function
    ln2: torch.nn.Linear
        Second linear layer in the custom head

    Methods:
    --------
    forward(x: torch.Tensor):
        Forward pass of the model
    '''

    def __init__(self, model_name, num_classes):
        '''
        Initialize the VisionTransformerWithCustomHead model with a pretrained Vision Transformer model and a custom head.

        Args:
        -----
        model_name: str
            Name of the pretrained Vision Transformer model
        num_classes: int
            Number of classes for classification
        '''
        super(VisionTransformerWithCustomHead, self).__init__()

        # Loading the pretrained ViT backbone without classification head from timm
        self.vit_model = timm.create_model(model_name, pretrained=True, num_classes=0)

        # Freezing the pretrained model parameters
        for param in self.vit_model.parameters():
            param.requires_grad = False

        # Custom MLP head
        self.ln1 = nn.Linear(768, 256)
        self.relu = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(256, num_classes)

    def forward(self, x):
        '''
        Forward pass of the model

        Args:
        -----
        x: torch.Tensor
            Input class token of the model (batch_size, embed_dim)

        Returns:
        --------
        output: torch.Tensor
            Output tensor from the model (batch_size, num_classes)
        '''
        x = self.vit_model(x)

        # Processing the class token through the custom head
        x = self.ln1(x)
        x = self.relu(x)
        output = self.ln2(x)
        
        return output