import numpy as np
import cv2
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms.functional import pad
from PIL import Image


def get_instance_view(
    map_view_img: np.ndarray,
    mask,
    mask_bg: bool = True,
    crop: bool = True,
    padding: int = 5,
) -> np.ndarray:
    """Get a view of the instance from the provided map view image and mask.

    This function extracts the view of an instance from a map view image
    using the provided mask. It can also crop, mask the background and
    apply padding to the view if specified.

    :param map_view_img: The map view image from which the instance view
        is to be extracted. This is a 3D numpy array with shape (height,
        width, channels).
    :param mask: The mask indicating the region of the instance in the
        map view image.
    :param mask_bg: If True, applies a black mask to the background of
        the image. Default is True.
    :param crop: If True, crops the image to the bounding box of the
        instance. Default is True.
    :param padding: The padding to be applied to the cropped view.
        Default is 5 pixels.
    :return: An image of the view of the instance.
    """
    coords = cv2.findNonZero(mask)
    # Get bounding box (x, y, width, height)
    x, y, w, h = cv2.boundingRect(coords)
    # Crop the image using the bounding box
    if mask_bg:
        image = cv2.bitwise_and(map_view_img, map_view_img, mask=mask)
    else:
        image = map_view_img
    if crop:
        image = image[
            max(y - padding, 0) : min(y + padding + h, map_view_img.shape[0]),
            max(x - padding, 0) : min(x + padding + w, map_view_img.shape[1]),
        ]
    return image


class SquarePad:
    """Class to apply square padding to an image.

    This class pads an image to make its dimensions square by adding
    equal padding to all sides.
    """

    def __call__(self, image) -> Tensor:
        """Apply square padding to the given image.

        :param image: The image to be padded. It is expected to be a PIL
            or torch.Tensor image.
        :return: The padded image as a Tensor.
        """
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return pad(image, padding, 0, "constant")


def preprocess_image(image0: np.ndarray, to_cuda: bool = True) -> Tensor:
    """Preprocess the input image for model inference.

    This function preprocesses the input image by applying square
    padding, resizing, normalization, and converting it to a tensor. It
    optionally moves the tensor to GPU.

    :param image0: The input image to be preprocessed. It is expected to
        be a numpy array with shape (height, width, channels).
    :param to_cuda: If True, moves the preprocessed image tensor to CUDA
        (GPU). Default is True.
    :return: The preprocessed image tensor.
    """
    image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
    image0_pil = Image.fromarray(image0)
    transform_val = Compose(
        [
            SquarePad(),
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    image0_tensor = transform_val(image0_pil)
    assert isinstance(image0_tensor, Tensor)
    if to_cuda:
        return image0_tensor[None, :].cuda()
    else:
        return image0_tensor[None, :]
