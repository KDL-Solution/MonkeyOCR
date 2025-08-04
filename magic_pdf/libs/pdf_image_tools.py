import cv2
import fitz
import hashlib
import numpy as np
from io import BytesIO
from PIL import Image

from magic_pdf.data.data_reader_writer import DataWriter
from magic_pdf.libs.commons import join_path


def _compute_sha256(input_string):
    hasher = hashlib.sha256()

    input_bytes = input_string.encode('utf-8')
    hasher.update(input_bytes)
    return hasher.hexdigest()


def cut_image(bbox: tuple, page_num: int, page: fitz.Page, return_path, imageWriter: DataWriter):

    filename = f'{page_num}_{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}'


    img_path = join_path(return_path, filename) if return_path is not None else None


    img_hash256_path = f'{_compute_sha256(img_path)}.jpg'


    rect = fitz.Rect(*bbox)

    zoom = fitz.Matrix(3, 3)

    pix = page.get_pixmap(clip=rect, matrix=zoom)

    byte_data = pix.tobytes(output='jpeg', jpg_quality=95)

    imageWriter.write(img_hash256_path, byte_data)

    return img_hash256_path


def cut_image_to_pil_image(bbox: tuple, page: fitz.Page, mode="pillow"):


    rect = fitz.Rect(*bbox)

    zoom = fitz.Matrix(3, 3)

    pix = page.get_pixmap(clip=rect, matrix=zoom)


    image_file = BytesIO(pix.tobytes(output='png'))

    pil_image = Image.open(image_file)
    if mode == "cv2":
        image_result = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    elif mode == "pillow":
        image_result = pil_image
    else:
        raise ValueError(f"mode: {mode} is not supported.")

    return image_result