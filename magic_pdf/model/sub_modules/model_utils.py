import time
import torch
from PIL import Image
from loguru import logger
from typing import Tuple, List, Dict, Any
from PIL import Image, ImageDraw, ImageFont

from magic_pdf.libs.clean_memory import clean_memory


def crop(
    image: Image.Image,
    res: Dict[str, Any],
    crop_paste_x: int = 0,
    crop_paste_y: int = 0,
) -> Tuple[Image.Image, List]:
    crop_xmin, crop_ymin = int(res["poly"][0]), int(res["poly"][1])
    crop_xmax, crop_ymax = int(res["poly"][4]), int(res["poly"][5])
    # Create a white background with an additional width and height of 50
    crop_new_width = crop_xmax - crop_xmin + crop_paste_x * 2
    crop_new_height = crop_ymax - crop_ymin + crop_paste_y * 2
    return_image = Image.new("RGB", (crop_new_width, crop_new_height), "white")

    # Crop image
    crop_box = (crop_xmin, crop_ymin, crop_xmax, crop_ymax)
    image_crop = image.crop(crop_box)
    return_image.paste(image_crop, (crop_paste_x, crop_paste_y))
    return_list = [crop_paste_x, crop_paste_y, crop_xmin, crop_ymin, crop_xmax, crop_ymax, crop_new_width, crop_new_height]
    return return_image, return_list


# Select regions for OCR / formula regions / table regions
def get_res_list_from_layout_res(layout_res):
    ocr_res_list = []
    table_res_list = []
    single_page_mfdetrec_res = []
    for res in layout_res:
        if int(res["category_id"]) in [13, 14]:
            single_page_mfdetrec_res.append({
                "bbox": [int(res["poly"][0]), int(res["poly"][1]),
                         int(res["poly"][4]), int(res["poly"][5])],
            })
        elif int(res["category_id"]) in [0, 1, 2, 4, 6, 7]:
            ocr_res_list.append(res)
        elif int(res["category_id"]) in [5]:
            table_res_list.append(res)
    return ocr_res_list, table_res_list, single_page_mfdetrec_res


def clean_vram(device, vram_threshold=8):
    total_memory = get_vram(device)
    if total_memory and total_memory <= vram_threshold:
        gc_start = time.time()
        clean_memory(device)
        gc_time = round(time.time() - gc_start, 2)
        logger.info(f"gc time: {gc_time}")


def get_vram(device):
    if torch.cuda.is_available() and device != "cpu":
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        return total_memory
    elif str(device).startswith("npu"):
        import torch_npu
        if torch_npu.npu.is_available():
            total_memory = torch_npu.npu.get_device_properties(device).total_memory / (1024 ** 3)
            return total_memory
    else:
        return None


### nested table 처리:
def mask(
    image: Image.Image,
    res_ls: List[Dict[str, Any]],
    bbox_color: str = "red",
    font_path: str = "arial.ttf",
    font_size: int = 32,
    text_color: str = "blue",
) -> Image.Image:
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    font = ImageFont.truetype(font_path, font_size)
    for idx, res in enumerate(
        res_ls,
        start=1,
    ):
        left = int(res["poly"][0])
        top = int(res["poly"][1])
        right = int(res["poly"][4])
        bottom = int(res["poly"][5])
        draw.rectangle(
            (left, top, right, bottom),
            fill=bbox_color,
        )
        draw.text(
            (left, top),
            f"[|ITEM-{idx:02d}|]",
            fill=text_color,
            font=font,
        )
    return image_copy
### : nested table 처리
