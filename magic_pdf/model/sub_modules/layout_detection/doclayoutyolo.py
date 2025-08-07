import time
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import List, Dict, Any
from loguru import logger
from doclayout_yolo import YOLOv10

from magic_pdf.data.dataset import BaseDataset


class DocLayoutYOLO(object):
    def __init__(self, weight, device):
        self.yolov10 = YOLOv10(weight)
        self.device = device

    def __call__(
        self,
        images: List[Image.Image],
        batch_size: int,
    ) -> List[List[Dict[str, Any]]]:
        model_out = []
        for index in range(0, len(images), batch_size):
            doclayout_yolo_res = [
                image_res.cpu()
                for image_res in self.yolov10.predict(
                    images[index : index + batch_size],
                    imgsz=1280,
                    conf=0.10,
                    iou=0.45,
                    verbose=False,
                    device=self.device,
                )
            ]
            for image_res in doclayout_yolo_res:
                layout_res = []
                for xyxy, conf, cla in zip(
                    image_res.boxes.xyxy,
                    image_res.boxes.conf,
                    image_res.boxes.cls,
                ):
                    xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                    new_item = {
                        "category_id": int(cla.item()),
                        "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                        "score": round(float(conf.item()), 3),
                    }
                    layout_res.append(new_item)
                model_out.append(layout_res)
        return model_out


def crop(
    input_res,
    input_pil_img,
    crop_paste_x=0,
    crop_paste_y=0,
):
    crop_xmin, crop_ymin = int(input_res['poly'][0]), int(input_res['poly'][1])
    crop_xmax, crop_ymax = int(input_res['poly'][4]), int(input_res['poly'][5])
    # Create a white background with an additional width and height of 50
    crop_new_width = crop_xmax - crop_xmin + crop_paste_x * 2
    crop_new_height = crop_ymax - crop_ymin + crop_paste_y * 2
    return_image = Image.new('RGB', (crop_new_width, crop_new_height), 'white')

    # Crop image
    crop_box = (crop_xmin, crop_ymin, crop_xmax, crop_ymax)
    cropped_img = input_pil_img.crop(crop_box)
    return_image.paste(cropped_img, (crop_paste_x, crop_paste_y))
    return_list = [crop_paste_x, crop_paste_y, crop_xmin, crop_ymin, crop_xmax, crop_ymax, crop_new_width, crop_new_height]
    return return_image, return_list


def layout_det_pre(
    dataset: BaseDataset,
) -> List[np.ndarray]:
    images = []
    for index in range(len(dataset)):
        page_data = dataset.get_page(index)
        img_dict = page_data.get_image()
        images.append(img_dict["img"])
    return images


def run_layout_det(
    images: List[np.ndarray],
    model: nn.Module,
    batch_size: int = 1
) -> List[List[Dict[str, Any]]]:
    layout_start_time = time.time()
    layout_images = [
        Image.fromarray(i) for i in images
    ]
    layout_det_out: List[List[Dict[str, Any]]] = model(
        layout_images,
        batch_size=batch_size,
    )  # Layout detection model inference.
    logger.info(
        f"layout time: {round(time.time() - layout_start_time, 2)}, image num: {len(images)}"
    )
    return layout_det_out


def layout_det_post(
    images: List[np.ndarray],
    layout_det_out,
) -> Dict[str, List[Any]]:
    new_images = []
    cat_ids = []
    page_indices = []
    for page_idx in range(len(images)):
        _layout_det_out: List[Dict[str, Any]] = layout_det_out[page_idx]
        image = Image.fromarray(images[page_idx])

        _new_images = []
        _cat_ids = []
        for layout_el in _layout_det_out:
            new_image, _ = crop(
                layout_el,
                image,
                crop_paste_x=50,
                crop_paste_y=50,
            )
            _new_images.append(new_image)
            _cat_ids.append(layout_el["category_id"])

        new_images.extend(_new_images)
        cat_ids.extend(_cat_ids)
        page_indices.append(len(new_images) - len(_new_images))
    return {
        "images": new_images,
        "category_ids": cat_ids,
        "page_indices": page_indices,
    }
