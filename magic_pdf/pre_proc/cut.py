import fitz
import hashlib
from typing import Tuple
from loguru import logger

from magic_pdf.config.ocr_content_type import ContentType
from magic_pdf.libs.commons import join_path
from magic_pdf.data.data_reader_writer import DataWriter


def _compute_sha256(
    input_string,
):
    hasher = hashlib.sha256()
    input_bytes = input_string.encode('utf-8')
    hasher.update(input_bytes)
    return hasher.hexdigest()


def _cut_image(
    bbox: Tuple,
    page_num: int,
    page: fitz.Page,
    return_path,
    imageWriter: DataWriter,
):
    filename = f'{page_num}_{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}'
    img_path = join_path(
        return_path,
        filename,
    ) if return_path is not None else None
    img_hash256_path = f'{_compute_sha256(img_path)}.jpg'

    rect = fitz.Rect(*bbox)
    zoom = fitz.Matrix(3, 3)
    pix = page.get_pixmap(clip=rect, matrix=zoom)
    byte_data = pix.tobytes(output='jpeg', jpg_quality=95)
    imageWriter.write(img_hash256_path, byte_data)
    return img_hash256_path


def _validate_bbox(
    bbox,
) -> bool:
    if any([bbox[0] >= bbox[2], bbox[1] >= bbox[3]]):
        logger.warning(f"image_bboxes: wrong box, {bbox}")
        return False
    return True


def _cut_image_and_table(
    spans,
    fitz_page,
    page_id,
    md5: str,
    image_writer,
):
    for span in spans:
        span_type = span["type"]
        if span_type == ContentType.Image:
            if not _validate_bbox(span["bbox"]) or not image_writer:
                continue
            span["image_path"] = _cut_image(
                span["bbox"],
                page_id,
                fitz_page,
                return_path=join_path(
                    md5,
                    "images",
                ),
                imageWriter=image_writer,
            )
    return spans
