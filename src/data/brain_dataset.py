from PIL import Image
from torchvision import datasets

class CustomBrainTumorDataset(datasets.ImageFolder):
    '''
    Custom dataset class for the Brain Tumor dataset.

    Attributes:
    -----------
    root: str
        Root directory of the dataset.
    transform: torchvision.transforms
        Transform to apply to the images.
    '''

    def __init__(self, root, transform=None):
        '''
        Constructor for the CustomBrainTumorDataset class.

        Args:
        -----
        root: str
            Root directory of the dataset.
        transform: torchvision.transforms
            Transform to apply to the images.
        '''
        super(CustomBrainTumorDataset, self).__init__(root, transform)

    def __getitem__(self, index):
        '''
        Get item method for the dataset class.

        Args:
        -----
        index: int
            Index of the item to retrieve.

        Returns:
        --------
        image: torch.Tensor
            Image tensor.
        target: int
            Target label.
        '''
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')

        # Apply transform if available
        if self.transform:
            image = self.transform(image)

        return image, target