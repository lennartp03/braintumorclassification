import torchvision.transforms.functional as TF

class CenterCropPercentage:
    '''
    Class to center crop an image by a certain percentage.

    Attributes:
    -----------
    crop_percentage: float
        Percentage to crop the image by.

    Methods:
    --------
    __call__(self, img: PIL.Image) -> PIL.Image:
        Apply the center crop to the image.
    '''

    def __init__(self, crop_percentage):
        '''
        Constructor for the CenterCropPercentage class.

        Args:
        -----
        crop_percentage: float
            Percentage to crop the image by.
        '''
        self.crop_percentage = crop_percentage

    def __call__(self, img):
        '''
        Apply the center crop to the image.

        Args:
        -----
        img: PIL.Image
            Image to crop.

        Returns:
        --------
        img_cropped: PIL.Image
            Cropped image.
        '''
        width, height = img.size
        crop_size = int(min(width, height) * self.crop_percentage / 100)

        # Calculate the top, left position of the crop
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2

        img_cropped = TF.crop(img, top, left, crop_size, crop_size)
        
        return img_cropped