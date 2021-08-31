from torchvision.transforms import functional as F
from PIL import Image

# for type hint
from typing import Union, Tuple, Callable, List
from collections.abc import Iterable
from torch import Tensor
from PIL.Image import Image as PILImage

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class CenterResizedCrop(object):
    """Crops the given PIL Image at the center. Then resize to desired shape.

    First, a largest possible center crop is performed at the center. Then,
    the cropped image is resized to the desired output size

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size: Union[int, Tuple[int, int]], interpolation: int = Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img: PILImage):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        image_width, image_height = img.size
        output_height, output_width = self.size

        image_ratio = image_width / image_height
        output_ratio = output_width / output_height

        if image_ratio >= output_ratio:
            crop_height = int(image_height)
            crop_width = int(image_height * output_ratio)
        else:
            crop_height = int(image_width / output_ratio)
            crop_width = int(image_width)

        cropped_img = F.center_crop(img, (crop_height, crop_width))

        return F.resize(cropped_img, size=self.size, interpolation=self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


__all__ = [
    "CenterResizedCrop",
]
