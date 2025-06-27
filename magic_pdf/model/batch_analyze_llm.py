import base64
import copy
import time
import re
import cv2
import numpy as np
import torch

from loguru import logger
from PIL import Image

from magic_pdf.config.constants import MODEL_NAME
from magic_pdf.config.ocr_content_type import CategoryId
from magic_pdf.config.chat_content_type import TaskInstructions, LoraType, LoraInstructions
from io import BytesIO, StringIO
from PIL import Image
from magic_pdf.model.sub_modules.model_utils import (
    clean_vram, crop_img)
from docling_core.types.doc.document import DocTagsDocument, DoclingDocument
YOLO_LAYOUT_BASE_BATCH_SIZE = 1

    
def sanitize_md(output):
        cleaned = re.match(r'<md>.*</md>', output, flags=re.DOTALL)
        if cleaned is None:
            return output.replace('<md>', '').replace('</md>', '').replace('md\n','').strip()
        return f"{cleaned[0].replace('<md>', '').replace('</md>', '').strip()}"
def sanitize_mf(output):
    cleaned = re.match(r'\$\$.*\$\$', output, flags=re.DOTALL)
    if cleaned is None:
        return output.replace('$$', '').strip()
    return f"{cleaned[0].replace('$$', '').strip()}"
def sanitize_html(output):
    otsl_match = re.search(r'<otsl>.*?</otsl>', output, flags=re.DOTALL)
    if otsl_match:
        try:
            otsl_text = otsl_match.group(0)
            stream = StringIO(otsl_text)
            table_tag = DocTagsDocument.from_doctags_and_image_pairs(stream, images=None)
            doc = DoclingDocument.load_from_doctags(table_tag)
            return doc.export_to_html()
        except Exception as e:
            return otsl_text.replace('<otsl>', '').replace('</otsl>', '').strip()
    
    cleaned = re.match(r'```html.*```', output, flags=re.DOTALL)
    if cleaned is None:
        return '<html>\n'+output.replace('```html','<html>').replace('```','</html>').strip()+'\n</html>'
    return f"{cleaned[0].replace('```html','<html>').replace('```','</html>').strip()}"

class LLMConfig:
    CATEGORY_MAPPING = {
        CategoryId.Title: {
            'task_instruction': TaskInstructions.TEXT,
            'lora_instruction': LoraInstructions.TEXT,
            'lora_type': LoraType.TEXT,
            'sanitizer': sanitize_md
        },
        CategoryId.Text: {
            'task_instruction': TaskInstructions.TEXT,
            'lora_instruction': LoraInstructions.TEXT,
            'lora_type': LoraType.TEXT,
            'sanitizer': sanitize_md
        },
        # 3 : {
        #     'task_instruction': TaskInstructions.TEXT,
        #     'lora_instruction': LoraInstructions.TEXT,
        #     'lora_type': LoraType.TEXT,
        #     'sanitizer': 'md'
        # },
        CategoryId.ImageCaption: {
            'task_instruction': TaskInstructions.TEXT,
            'lora_instruction': LoraInstructions.TEXT,
            'lora_type': LoraType.TEXT,
            'sanitizer': sanitize_md
        },
        CategoryId.TableBody: {
            'task_instruction': TaskInstructions.TABLE,
            'lora_instruction': LoraInstructions.TABLE,
            'lora_type': LoraType.TABLE,
            'sanitizer': sanitize_html
        },
        CategoryId.TableCaption: {
            'task_instruction': TaskInstructions.TEXT,
            'lora_instruction': LoraInstructions.TEXT,
            'lora_type': LoraType.TEXT,
            'sanitizer': sanitize_md
        },
        CategoryId.TableFootnote: {
            'task_instruction': TaskInstructions.TEXT,
            'lora_instruction': LoraInstructions.TEXT,
            'lora_type': LoraType.TEXT,
            'sanitizer': sanitize_md
        },
        CategoryId.InterlineEquation_Layout: {
            'task_instruction': TaskInstructions.FORMULA,
            'lora_instruction': LoraInstructions.BASE, 
            'lora_type': LoraType.BASE,
            'sanitizer': sanitize_mf
        },
        # 9 : {
        #     'task_instruction': TaskInstructions.TEXT,
        #     'lora_instruction': LoraInstructions.BASE,
        #     'lora_type': LoraType.BASE,
        #     'sanitizer': sanitize_md
        # },
        CategoryId.InterlineEquation_YOLO: {
            'task_instruction': TaskInstructions.FORMULA,
            'lora_instruction': LoraInstructions.BASE,
            'lora_type': LoraType.BASE,
            'sanitizer': sanitize_mf
        },
        CategoryId.ImageFootnote: {
            'task_instruction': TaskInstructions.TEXT,
            'lora_instruction': LoraInstructions.TEXT,
            'lora_type': LoraType.TEXT,
            'sanitizer': sanitize_md
        },
    }
    
    @classmethod
    def get_instruction(cls, category_id, version='task'):
        """CategoryId와 version에 따른 instruction 반환"""
        mapping = cls.CATEGORY_MAPPING.get(category_id, {})
        
        if version == 'lora':  
            return mapping.get('lora_instruction')
        else: 
            return mapping.get('task_instruction')
    
    @classmethod
    def get_lora_type(cls, category_id):
        """CategoryId에 해당하는 LoRA 타입 반환"""
        return cls.CATEGORY_MAPPING.get(category_id, {}).get('lora_type', LoraType.BASE)
    
    @classmethod
    def get_sanitizer(cls, category_id):
        """CategoryId에 해당하는 sanitizer 타입 반환"""
        return cls.CATEGORY_MAPPING.get(category_id, {}).get('sanitizer', sanitize_md)
    
    @classmethod
    def is_supported(cls, category_id):
        """지원되는 CategoryId인지 확인"""
        return category_id in cls.CATEGORY_MAPPING

    
    
    
class BatchAnalyzeLLM:
    def __init__(self, model, backend='lmdeploy'):
        self.model = model
        self.backend = backend

    def __call__(self, images: list):
        images_layout_res = []

        layout_start_time = time.time()
        if self.model.layout_model_name == MODEL_NAME.DocLayout_YOLO:
            # doclayout_yolo
            layout_images = []
            modified_images = []
            for image_index, image in enumerate(images):
                pil_img = Image.fromarray(image)
                layout_images.append(pil_img)

            images_layout_res += self.model.layout_model.batch_predict(
                # layout_images, self.batch_ratio * YOLO_LAYOUT_BASE_BATCH_SIZE
                layout_images, YOLO_LAYOUT_BASE_BATCH_SIZE
            )

            for image_index, useful_list in modified_images:
                for res in images_layout_res[image_index]:
                    for i in range(len(res['poly'])):
                        if i % 2 == 0:
                            res['poly'][i] = (
                                res['poly'][i] - useful_list[0] + useful_list[2]
                            )
                        else:
                            res['poly'][i] = (
                                res['poly'][i] - useful_list[1] + useful_list[3]
                            )
        logger.info(
            f'layout time: {round(time.time() - layout_start_time, 2)}, image num: {len(images)}'
        )

        clean_vram(self.model.device, vram_threshold=8)

        llm_ocr_start = time.time()
        new_images_all = []
        cids_all = []
        page_idxs = []
        for index in range(len(images)):
            layout_res = images_layout_res[index]
            pil_img = Image.fromarray(images[index])
            new_images = []
            cids = []
            for res in layout_res:
                new_image, useful_list = crop_img(
                    res, pil_img, crop_paste_x=50, crop_paste_y=50
                )
                # save new_image
                new_image = new_image.convert('RGB')
                new_image.save("./temp_image.jpg", format='JPEG')
                new_images.append(new_image)
                cids.append(res['category_id'])
            
            new_images_all.extend(new_images)
            cids_all.extend(cids)
            page_idxs.append(len(new_images_all) - len(new_images))
        logger.info('VLM OCR start...')
        ocr_result = self.batch_llm_ocr(new_images_all, cids_all)
        print(images_layout_res)
        for index in range(len(images)):
            ocr_results = []
            layout_res = images_layout_res[index]
            for i in range(len(layout_res)):
                res = layout_res[i]
                ocr = ocr_result[page_idxs[index]+i]
                # ocr = self.llm_ocr(new_image, res['category_id'])
                if res['category_id'] in [
                    CategoryId.InterlineEquation_Layout, 
                    CategoryId.InterlineEquation_YOLO
                ]:
                    temp_res = copy.deepcopy(res)
                    temp_res['category_id'] = CategoryId.InterlineEquation_YOLO
                    temp_res['score'] = 1.0
                    temp_res['latex'] = ocr
                    ocr_results.append(temp_res)
                elif res['category_id'] in [
                    CategoryId.Title, 
                    CategoryId.Text, 
                    CategoryId.Abandon, 
                    CategoryId.ImageCaption, 
                    CategoryId.TableCaption, 
                    CategoryId.TableFootnote, 
                    CategoryId.ImageFootnote
                ]:
                    temp_res = copy.deepcopy(res)
                    temp_res['category_id'] = CategoryId.OcrText
                    temp_res['score'] = 1.0
                    temp_res['text'] = ocr
                    ocr_results.append(temp_res)
                elif res['category_id'] == CategoryId.TableBody:
                    res['score'] = 1.0
                    res['html'] = ocr
            layout_res.extend(ocr_results)
            logger.info(f'OCR processed images / total images: {index+1} / {len(images)}')
        logger.info(
            f'llm ocr time: {round(time.time() - llm_ocr_start, 2)}, image num: {len(images)}'
        )

        return images_layout_res

    def batch_llm_ocr(self, images, cat_ids,max_batch_size=8):

        assert len(images) == len(cat_ids)
                    
        new_images = []
        messages = []
        model_types = []
        ignore_idx = []
        outs = []
        if self.backend in ['vllm', 'lmdeploy', 'openai_api']:
            for i in range(len(images)):
                if not LLMConfig.is_supported(cat_ids[i]):
                    ignore_idx.append(i)
                    continue
                new_images.append(images[i])
                messages.append(LLMConfig.get_instruction(cat_ids[i]))
            out = self.model.chat_model.batch_inference(new_images, messages)
            outs.extend(out)
        elif self.backend in ['vllm_api']:
            for i in range(len(images)):
                if not LLMConfig.is_supported(cat_ids[i]):
                    ignore_idx.append(i)
                    continue
                new_images.append(images[i])
                messages.append(LLMConfig.get_instruction(cat_ids[i], version='lora'))
                lora_type = LLMConfig.get_lora_type(cat_ids[i])
                model_types.append(lora_type)
            out = self.model.chat_model.batch_inference(new_images, messages, model_types=model_types)
            outs.extend(out)
        else:
            buffer = BytesIO()
            for i in range(len(images)):
                if not LLMConfig.is_supported(cat_ids[i]):
                    ignore_idx.append(i)
                    continue
                images[i].save(buffer, format='JPEG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                messages.append(
                    [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": "data:image/jpeg;base64," + image_base64,
                            },
                            {"type": "text", "text": "{}".format(LLMConfig.get_instruction(cat_ids[i]))}
                        ],
                    },]
                )
                buffer.seek(0)
                buffer.truncate(0)
                # if len(messages) == max_batch_size or i == len(images) - 1:
            outs.extend(self.model.llm_model.batch_inference(messages))
        for j in ignore_idx:
            outs.insert(j, '')
        messages.clear()
        ignore_idx.clear()
        for j in range(len(outs)):
            if LLMConfig.is_supported(cat_ids[j]):
                sanitizer = LLMConfig.get_sanitizer(cat_ids[j])
                outs[j] = sanitizer(outs[j])
        return outs