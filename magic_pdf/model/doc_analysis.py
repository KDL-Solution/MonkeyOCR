import time
import copy
import time
import numpy as np
from loguru import logger
from loguru import logger
from PIL import Image
from typing import List, Dict, Any
from PIL import Image

from magic_pdf.data.dataset import Dataset
from magic_pdf.libs.clean_memory import clean_memory
from magic_pdf.operators.pipe_result import InferenceResult
from magic_pdf.config.ocr_content_type import CategoryId
from magic_pdf.config.prompts import PromptConfig
from magic_pdf.model.sub_modules.model_utils import (
    clean_vram,
    crop_img,
)
from magic_pdf.model.monkeyocr import MonkeyOCR

YOLO_LAYOUT_BASE_BATCH_SIZE = 1


class InferenceBatch:
    def __init__(
        self,
        monkeyocr: MonkeyOCR,
    ):
        self.monkeyocr = monkeyocr

    def __call__(
        self,
        images: List[np.ndarray],
    ) -> List:
        layout_start_time = time.time()
        layout_images = [
            Image.fromarray(i) for i in images
        ]
        layout_det_out: List[List[Dict[str, Any]]] = self.monkeyocr.layout_det(
            layout_images,
            batch_size=YOLO_LAYOUT_BASE_BATCH_SIZE,
        )  # Layout detection model inference.
        logger.info(
            f"layout time: {round(time.time() - layout_start_time, 2)}, image num: {len(images)}"
        )

        clean_vram(
            self.monkeyocr.device,
            vram_threshold=8,
        )

        llm_start = time.time()
        new_images_all = []
        cat_ids = []
        page_indices = []
        for idx in range(len(images)):
            _layout_det_out: List[Dict[str, Any]] = layout_det_out[idx]
            pil_img = Image.fromarray(images[idx])

            new_images = []
            cat_ids_ = []
            for layout_el in _layout_det_out:
                new_image, _ = crop_img(
                    layout_el,
                    pil_img,
                    crop_paste_x=50,
                    crop_paste_y=50,
                )
                new_images.append(new_image)
                cat_ids_.append(layout_el["category_id"])

            new_images_all.extend(new_images)
            cat_ids.extend(cat_ids_)
            page_indices.append(len(new_images_all) - len(new_images))

        logger.info("LLM inference start...")
        llm_out: List[str] = self.llm_infer_batch(
            images=new_images_all,
            cat_ids=cat_ids,
        )  # LLM inference.
        ### layout detection 출력에 LLM 출력을 추가:
        for idx1 in range(len(images)):
            ocr_results = []
            _layout_det_out = layout_det_out[idx1]
            for idx2 in range(len(_layout_det_out)):
                layout_el = _layout_det_out[idx2]
                _llm_out = llm_out[page_indices[idx1] + idx2]

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
            logger.info(f"LLM processed images: {idx1 + 1} / {len(images)}")
        logger.info(
            f"llm ocr time: {round(time.time() - llm_start, 2)}, image num: {len(images)}"
        )
        ### : layout detection 출력에 LLM 출력을 추가
        return layout_det_out

    def llm_infer_batch(
        self,
        images,
        cat_ids,
    ) -> List[str]:
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

        out = self.monkeyocr.llm(
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


def doc_analyze(
    dataset: Dataset,
    monkeyocr: MonkeyOCR,
) -> InferenceResult:
    inference_batch = InferenceBatch(
        monkeyocr=monkeyocr,
    )

    model_json = []
    doc_analyze_start = time.time()

    images = []
    for index in range(len(dataset)):
        page_data = dataset.get_page(index)
        img_dict = page_data.get_image()
        images.append(img_dict["img"])

    analyze_result = inference_batch(
        images,
    )

    for index in range(len(dataset)):
        page_data = dataset.get_page(index)
        img_dict = page_data.get_image()
        page_width = img_dict["width"]
        page_height = img_dict["height"]
        result = analyze_result.pop(0)

        page_info = {
            "page_no": index,
            "height": page_height,
            "width": page_width,
        }
        page_dict = {
            "layout_dets": result,
            "page_info": page_info,
        }
        model_json.append(page_dict)

    gc_start = time.time()
    clean_memory(monkeyocr.device)
    gc_time = round(time.time() - gc_start, 2)
    logger.info(f"gc time: {gc_time}")

    doc_analyze_time = round(time.time() - doc_analyze_start, 2)
    doc_analyze_speed = round(len(dataset) / doc_analyze_time, 2)
    logger.info(
        f"doc analyze time: {round(time.time() - doc_analyze_start, 2)},"
        f"speed: {doc_analyze_speed} pages/second"
    )
    return InferenceResult(
        model_json,
        dataset,
    )
