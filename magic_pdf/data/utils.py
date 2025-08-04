import fitz
import numpy as np
from loguru import logger


def ImportPIL(f):
    try:
        import PIL  # noqa: F401
    except ImportError:
        logger.error('Pillow not installed, please install by pip.')
        exit(1)
    return f


@ImportPIL
def fitz_page_to_image(
    page,
    dpi=200,
) -> dict:
    """Convert fitz.Document to image, Then convert the image to numpy array.

    Args:
        doc (_type_): pymudoc page
        dpi (int, optional): reset the dpi of dpi. Defaults to 200.

    Returns:
        dict:  {'img': numpy array, 'width': width, 'height': height }
    """
    from PIL import Image
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pm = page.get_pixmap(matrix=mat, alpha=False)

    # If the width or height exceeds 4500 after scaling, do not scale further.
    if pm.width > 4500 or pm.height > 4500:
        pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

    img = Image.frombytes('RGB', (pm.width, pm.height), pm.samples)
    img = np.array(img)

    img_dict = {'img': img, 'width': pm.width, 'height': pm.height}
    return img_dict
