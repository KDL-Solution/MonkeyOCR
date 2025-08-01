from PIL import Image
from typing import List, Dict, Any
from doclayout_yolo import YOLOv10


class DocLayoutYOLO(object):
    def __init__(self, weight, device):
        self.yolov10 = YOLOv10(weight)
        self.device = device

    # def predict(self, image):
    #     layout_res = []
    #     doclayout_yolo_res = self.yolov10.predict(
    #         image,
    #         imgsz=1280,
    #         conf=0.10,
    #         iou=0.45,
    #         verbose=False, device=self.device
    #     )[0]
    #     for xyxy, conf, cla in zip(
    #         doclayout_yolo_res.boxes.xyxy.cpu(),
    #         doclayout_yolo_res.boxes.conf.cpu(),
    #         doclayout_yolo_res.boxes.cls.cpu(),
    #     ):
    #         xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
    #         new_item = {
    #             "category_id": int(cla.item()),
    #             "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
    #             "score": round(float(conf.item()), 3),
    #         }
    #         layout_res.append(new_item)
    #     return layout_res

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
