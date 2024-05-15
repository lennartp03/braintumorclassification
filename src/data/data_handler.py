from torch.utils.data import DataLoader, random_split

class BrainTumorDataHandler():
    '''
    A class to handle and prepare the tumor dataset for training, validation and testing.

    Attributes:
    -----------
    train_data : torch.utils.data.Dataset
        Training dataset.
    test_data : torch.utils.data.Dataset
        Testing dataset.
    train_val_split : float
        The ratio to split the training data into training and validation.
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Number of subprocesses to use for data loading.
    pin_memory : bool
        Whether to copy tensors into CUDA pinned memory.

    Methods:
    --------
    _prepare_dataset():
        Split the training data into training and validation sets and prepare the data loaders.
    get_loaders():
        Return the training, validation and testing data loaders.
    '''
    
    def __init__(self, train_dataset, test_dataset, train_val_split, batch_size, num_workers, pin_memory):
        '''
        Initialize the BrainTumorData class.

        Args:
        -----
        train_dataset : torch.utils.data.Dataset
            Training dataset.
        test_dataset : torch.utils.data.Dataset
            Testing dataset.
        train_val_split : float
            The ratio to split the training data into training and validation.
        batch_size : int
            Number of samples per batch.
        num_workers : int
            Number of subprocesses to use for data loading.
        pin_memory : bool
            Whether to copy tensors into CUDA pinned memory.
        '''
        self.train_data = train_dataset
        self.test_data = test_dataset
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self._prepare_dataset()

    def _prepare_dataset(self):
        '''
        Split the training data into training and validation sets and prepare the data loaders.
        '''

        train_size = int(self.train_val_split * len(self.train_data))
        val_size = len(self.train_data) - train_size
        self.train_data, self.val_data= random_split(self.train_data, [train_size, val_size])

        self.train_loader = DataLoader(self.train_data, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)
        self.val_loader = DataLoader(self.val_data, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)
        self.test_loader = DataLoader(self.test_data, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def get_loaders(self):
        '''
        Return the training, validation and testing data loaders.

        Returns:
        --------
        _ : tuple
            Tuple of training, validation and testing data loaders.
        '''
        
        return self.train_loader, self.val_loader, self.test_loader