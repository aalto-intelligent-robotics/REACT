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
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return pad(image, padding, 0, "constant")


def preprocess_image(image0: np.ndarray, to_cuda: bool = True) -> Tensor:
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
