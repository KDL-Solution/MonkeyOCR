from loguru import logger

from magic_pdf.config.ocr_content_type import ContentType
from magic_pdf.libs.commons import join_path
from magic_pdf.libs.pdf_image_tools import cut_image


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
    pdf_bytes_md5,
    image_writer,
):
    for span in spans:
        span_type = span["type"]
        if span_type == ContentType.Image:
            if not _validate_bbox(span["bbox"]) or not image_writer:
                continue
            span["image_path"] = cut_image(
                span["bbox"],
                page_id,
                fitz_page,
                return_path=join_path(
                    pdf_bytes_md5,
                    type="images",
                ),
                imageWriter=image_writer,
            )
        # elif span_type == ContentType.Table:
        #     if not check_img_bbox(span["bbox"]) or not image_writer:
        #         continue
        #     span["image_path"] = cut_image(
        #         span["bbox"],
        #         page_id,
        #         fitz_page,
        #         return_path=
        #         join_path(
        #             pdf_bytes_md5,
        #             type="tables",
        #         ),
        #         imageWriter=image_writer,
        #     )
    return spans
