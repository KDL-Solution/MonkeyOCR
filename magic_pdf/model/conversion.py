import time
import copy
import numpy as np
from loguru import logger
from PIL import Image
from typing import List, Dict, Any
from PIL import Image

from magic_pdf.data.dataset import BaseDataset
from magic_pdf.data.data_reader_writer import DataWriter
from magic_pdf.libs.clean_memory import clean_memory
from magic_pdf.operators.conversion_result import IntermediateConversionResult
from magic_pdf.config.ocr_content_type import CategoryId
from magic_pdf.config.prompts import PromptConfig
from magic_pdf.model.sub_modules.model_utils import clean_vram
from magic_pdf.model.monkeyocr import MonkeyOCR
from magic_pdf.model.sub_modules.layout_detection.doclayoutyolo import (
    layout_det_pre,
    run_layout_det,
    layout_det_post,
)


def run_llm(
    images,
    cat_ids,
    model,
) -> List[str]:
    logger.info("LLM inference start...")

    assert len(images) == len(cat_ids)

    new_images = []
    user_prompts = []
    model_names = []
    ignore_idx = []
    outs = []
    for i in range(len(images)):
        new_images.append(
            images[i]
        )

        cat_id = cat_ids[i]
        if not PromptConfig.is_supported_category(cat_id):
            ignore_idx.append(i)
            continue

        user_prompts.append(
            PromptConfig.get_user_prompt(
                cat_id,
            )
        )
        model_names.append(
            PromptConfig.get_model_name(
                cat_id,
            )
        )

    out = model(
        images=new_images,
        user_prompts=user_prompts,
        model_names=model_names,
    )
    outs.extend(out)

    for j in ignore_idx:
        outs.insert(j, "")
    user_prompts.clear()
    ignore_idx.clear()
    for j in range(len(outs)):
        if PromptConfig.is_supported_category(cat_ids[j]):
            sanitizer = PromptConfig.get_sanitizer(cat_ids[j])
            outs[j] = sanitizer(outs[j])
    return outs


def llm_post(
    layout_det_out,
    llm_out,
    page_indices,
):
    """layout detection 출력에 LLM 출력을 추가
    """
    for page_idx in range(len(page_indices)):
        ocr_results = []
        _layout_det_out = layout_det_out[page_idx]
        for idx2 in range(len(_layout_det_out)):
            layout_el = _layout_det_out[idx2]
            _llm_out = llm_out[page_indices[page_idx] + idx2]

            if layout_el["category_id"] in [
                CategoryId.InterlineEquation_Layout, 
                CategoryId.InterlineEquation_YOLO,
            ]:
                layout_el_copy = copy.deepcopy(layout_el)
                layout_el_copy["category_id"] = CategoryId.InterlineEquation_YOLO
                layout_el_copy["score"] = 1.
                layout_el_copy["latex"] = _llm_out
                ocr_results.append(layout_el_copy)
            elif layout_el["category_id"] in [
                CategoryId.Title, 
                CategoryId.Text, 
                CategoryId.Abandon, 
                CategoryId.ImageCaption, 
                CategoryId.TableCaption, 
                CategoryId.TableFootnote, 
                CategoryId.ImageFootnote
            ]:
                layout_el_copy = copy.deepcopy(layout_el)
                layout_el_copy["category_id"] = CategoryId.OcrText
                layout_el_copy["score"] = 1.
                layout_el_copy["text"] = _llm_out
                ocr_results.append(layout_el_copy)
            elif layout_el["category_id"] == CategoryId.TableBody:
                layout_el["score"] = 1.
                layout_el["html"] = _llm_out
        _layout_det_out.extend(ocr_results)
        logger.info(f"LLM processed images: {page_idx + 1} / {len(page_indices)}")
    # logger.info(
    #     f"llm ocr time: {round(time.time() - llm_start, 2)}, image num: {len(page_indices)}"
    # )
    return layout_det_out


def convert(
    dataset: BaseDataset,
    image_writer: DataWriter,
    monkeyocr: MonkeyOCR,
) -> IntermediateConversionResult:
    conv_start = time.time()

    ### Layout detection:
    images = layout_det_pre(
        dataset,
    )
    layout_det_out = run_layout_det(
        images,
        model=monkeyocr.layout_det,
    )
    clean_vram(
        monkeyocr.device,
        vram_threshold=8,
    )
    layout_det_post_out = layout_det_post(
        images,
        layout_det_out,
    )
    ### : Layout detection

    ### LLM:
    llm_out = run_llm(
        layout_det_post_out["images"],
        cat_ids=layout_det_post_out["category_ids"],
        model=monkeyocr.llm,
    )  # LLM inference.
    llm_post_out = llm_post(
        layout_det_out=layout_det_out,
        llm_out=llm_out,
        page_indices=layout_det_post_out["page_indices"],
    )
    ### : LLM

    model_json = []
    for index in range(len(dataset)):
        page_data = dataset.get_page(index)
        img_dict = page_data.get_image()
        model_json.append(
            {
                "layout_dets": llm_post_out.pop(0),
                "page_info": {
                    "page_no": index,
                    "height": img_dict["height"],  # page height
                    "width": img_dict["width"],  # page width
                },
            }
        )

    gc_start = time.time()
    clean_memory(monkeyocr.device)
    gc_time = round(time.time() - gc_start, 2)
    logger.info(f"gc time: {gc_time}")

    conv_time = round(time.time() - conv_start, 2)
    conv_speed = round(len(dataset) / conv_time, 2)
    logger.info(
        f"document conversion time: {round(time.time() - conv_start, 2)},"
        f"speed: {conv_speed} pages/second"
    )
    ### Relation prediction:
    return IntermediateConversionResult(
        model_json,
        dataset,
    ).make_conversion_result(
        image_writer=image_writer,
        monkeyocr=monkeyocr,
    )
    ### : Relation prediction
