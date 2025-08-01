import copy
import time
import re
import numpy as np
from loguru import logger
from PIL import Image
from typing import List, Dict, Any
from io import StringIO
from PIL import Image
from docling_core.types.doc.document import DocTagsDocument, DoclingDocument

from magic_pdf.config.ocr_content_type import CategoryId
from magic_pdf.config.prompts import (
    TaskInstructions,
    LoRAType,
    LoRAInstructions,
)
from magic_pdf.model.sub_modules.model_utils import (
    clean_vram,
    crop_img,
)
from magic_pdf.model.monkeyocr import MonkeyOCR

YOLO_LAYOUT_BASE_BATCH_SIZE = 1


def sanitize_md(output):
    cleaned = re.match(r"<md>.*</md>", output, flags=re.DOTALL)
    if cleaned is None:
        return output.replace("<md>", "").replace("</md>", "").replace("md\n","").strip()
    return f"""{cleaned[0].replace("<md>", "").replace("</md>", "").strip()}"""


def sanitize_math_formula(output):
    cleaned = re.match(r"\$\$.*\$\$", output, flags=re.DOTALL)
    if cleaned is None:
        return output.replace("$$", "").strip()
    return f"""{cleaned[0].replace("$$", "").strip()}"""


def sanitize_html(output):
    otsl_match = re.search(r"<otsl>.*?</otsl>", output, flags=re.DOTALL)
    if otsl_match:
        try:
            otsl_text = otsl_match.group(0)
            stream = StringIO(otsl_text)
            table_tag = DocTagsDocument.from_doctags_and_image_pairs(stream, images=None)
            doc = DoclingDocument.load_from_doctags(table_tag)
            table_html = []
            for table in doc.tables:
                table_html.append(table.export_to_html(doc=doc))
            if len(table_html) == 0:
                return otsl_text.replace("<otsl>", "").replace("</otsl>", "").strip()
            # If there are tables, return the first table"s HTML
            table_html = "\n".join(table_html)
            return table_html.replace("<otsl>", "").replace("</otsl>", "").strip()
        except Exception as e:
            return otsl_text.replace("<otsl>", "").replace("</otsl>", "").strip()
    
    cleaned = re.match(r"```html.*```", output, flags=re.DOTALL)
    if cleaned is None:
        return "<html>\n"+output.replace("```html","<html>").replace("```","</html>").strip()+"\n</html>"
    return f"""{cleaned[0].replace("```html","<html>").replace("```","</html>").strip()}"""


class LLMConfig:
    CATEGORY_MAPPING = {
        CategoryId.Title: {
            "task_instruction": TaskInstructions.TEXT,
            "LoRA_instruction": LoRAInstructions.TEXT,
            "LoRA_type": LoRAType.BASE,
            "sanitizer": sanitize_md
        },
        CategoryId.Text: {
            "task_instruction": TaskInstructions.TEXT,
            "LoRA_instruction": LoRAInstructions.TEXT,
            "LoRA_type": LoRAType.BASE,
            "sanitizer": sanitize_md
        },
        CategoryId.Abandon: {
            "task_instruction": TaskInstructions.TEXT,
            "LoRA_instruction": LoRAInstructions.TEXT,
            "LoRA_type": LoRAType.BASE,
            "sanitizer": sanitize_md
        },
        CategoryId.ImageBody: {
            "task_instruction": TaskInstructions.Image,
            "LoRA_instruction": LoRAInstructions.Image,
            "LoRA_type": LoRAType.BASE,
            "sanitizer": sanitize_md
        },
        CategoryId.ImageCaption: {
            "task_instruction": TaskInstructions.TEXT,
            "LoRA_instruction": LoRAInstructions.TEXT,
            "LoRA_type": LoRAType.BASE,
            "sanitizer": sanitize_md
        },
        CategoryId.TableBody: {
            "task_instruction": TaskInstructions.TABLE,
            "LoRA_instruction": LoRAInstructions.TABLE,
            "LoRA_type": LoRAType.TABLE,
            "sanitizer": sanitize_html
        },
        CategoryId.TableCaption: {
            "task_instruction": TaskInstructions.TEXT,
            "LoRA_instruction": LoRAInstructions.TEXT,
            "LoRA_type": LoRAType.BASE,
            "sanitizer": sanitize_md
        },
        CategoryId.TableFootnote: {
            "task_instruction": TaskInstructions.TEXT,
            "LoRA_instruction": LoRAInstructions.TEXT,
            "LoRA_type": LoRAType.BASE,
            "sanitizer": sanitize_md
        },
        CategoryId.InterlineEquation_Layout: {
            "task_instruction": TaskInstructions.FORMULA,
            "sanitizer": sanitize_math_formula
        },
        CategoryId.InterlineEquation_YOLO: {
            "task_instruction": TaskInstructions.FORMULA,
            "sanitizer": sanitize_math_formula
        },
        CategoryId.ImageFootnote: {
            "task_instruction": TaskInstructions.TEXT,
            "LoRA_instruction": LoRAInstructions.TEXT,
            "LoRA_type": LoRAType.BASE,
            "sanitizer": sanitize_md
        },
    }

    @classmethod
    def get_instruction(cls, category_id, version="task"):
        """CategoryId와 version에 따른 instruction 반환"""
        mapping = cls.CATEGORY_MAPPING.get(category_id, {})

        if version == "LoRA":  
            return mapping.get("LoRA_instruction")
        else: 
            return mapping.get("task_instruction")

    @classmethod
    def get_LoRA_type(cls, category_id):
        """CategoryId에 해당하는 LoRA 타입 반환"""
        return cls.CATEGORY_MAPPING.get(category_id, {}).get("LoRA_type", LoRAType.BASE)

    @classmethod
    def get_sanitizer(cls, category_id):
        """CategoryId에 해당하는 sanitizer 타입 반환"""
        return cls.CATEGORY_MAPPING.get(category_id, {}).get("sanitizer", sanitize_md)

    @classmethod
    def is_supported(cls, category_id):
        """지원되는 CategoryId인지 확인"""
        return category_id in cls.CATEGORY_MAPPING

    @classmethod
    def is_LoRA_supported(cls, category_id):
        """CategoryId에 해당하는 LoRA가 지원되는지 확인"""
        return cls.get_LoRA_type(category_id) != LoRAType.BASE and cls.is_supported(category_id)


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
        # max_batch_size=8,  # 미사용.
    ) -> List[str]:
        assert len(images) == len(cat_ids)

        new_images = []
        user_prompts = []
        model_types = []
        ignore_idx = []
        outs = []
        for i in range(len(images)):
            new_images.append(
                images[i]
            )

            cat_id = cat_ids[i]

            if not LLMConfig.is_supported(cat_id):
                ignore_idx.append(i)
                continue

            if LLMConfig.is_LoRA_supported(cat_id):
                user_prompts.append(
                    LLMConfig.get_instruction(
                        cat_id,
                        version="LoRA",
                    )
                )
                LoRA_type = LLMConfig.get_LoRA_type(cat_id)
                model_types.append(LoRA_type)
            else:
                user_prompts.append(
                    LLMConfig.get_instruction(
                        cat_id,
                    ),
                )
                model_types.append(LoRAType.BASE)

        out = self.monkeyocr.llm(
            images=new_images,
            user_prompts=user_prompts,
            model_types=model_types,
        )
        outs.extend(out)

        for j in ignore_idx:
            outs.insert(j, "")
        user_prompts.clear()
        ignore_idx.clear()
        for j in range(len(outs)):
            if LLMConfig.is_supported(cat_ids[j]):
                sanitizer = LLMConfig.get_sanitizer(cat_ids[j])
                outs[j] = sanitizer(outs[j])
        return outs
