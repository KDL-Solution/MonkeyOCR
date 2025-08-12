import torch
from collections import defaultdict
from typing import List, Dict
from transformers import LayoutLMv3ForTokenClassification


class RelationPrediction(object):
    def __init__(
        self,
        model: LayoutLMv3ForTokenClassification,
    ) -> None:
        self.model = model

    def _normalize(
        self,
        bboxes: List[List[int]],
        width: int,
        height: int,
        scale: int = 1000
    ) -> List[List[int]]:
        new_bboxes = []
        for left, top, right, bottom in bboxes:
            left = round(max(0, left) / width * scale)
            top = round(max(0, top) / height * scale)
            right = round(min(width, right) / width * scale)
            bottom = round(min(height, bottom) / height * scale)

            assert (
                0 <= left <= right <= scale and 0 <= top <= bottom <= scale
            ), f"Invalid box. right: {right}, left: {left}, bottom: {bottom}, top: {top}"  # noqa: E126, E121
            new_bboxes.append([left, top, right, bottom])
        return new_bboxes

    def rel_pred_pre(
        self,
        bboxes: List[List[int]],
        width: int,
        height: int,
        cls_token_id = 0,
        unk_token_id = 3,
        eos_token_id = 2,
    ) -> Dict[str, torch.Tensor]:
        bboxes = self._normalize(
            bboxes=bboxes,
            width=width,
            height=height,
        )
        bbox = [[0, 0, 0, 0]] + bboxes + [[0, 0, 0, 0]]
        input_ids = [cls_token_id] + [unk_token_id] * len(bboxes) + [eos_token_id]
        attention_mask = [1] + [1] * len(bboxes) + [1]
        inputs = {
            "bbox": torch.tensor([bbox]),
            "attention_mask": torch.tensor([attention_mask]),
            "input_ids": torch.tensor([input_ids]),
        }
        new_inputs = {}
        for k, v in inputs.items():
            v = v.to(self.model.device)
            if torch.is_floating_point(v):
                v = v.to(self.model.dtype)
            new_inputs[k] = v
        return new_inputs

    def rel_pred_post(
        self,
        logits: torch.Tensor,
        length: int,
    ) -> List[int]:
        """
        parse logits to orders

        :param logits: logits from model
        :param length: input length
        :return: orders
        """
        logits = logits[1 : length + 1, :length]
        orders = logits.argsort(descending=False).tolist()
        ret = [o.pop() for o in orders]
        while True:
            order_to_idxes = defaultdict(list)
            for idx, order in enumerate(ret):
                order_to_idxes[order].append(idx)
            # filter idxes len > 1
            order_to_idxes = {k: v for k, v in order_to_idxes.items() if len(v) > 1}
            if not order_to_idxes:
                break
            # filter
            for order, idxes in order_to_idxes.items():
                # find original logits of idxes
                idxes_to_logit = {}
                for idx in idxes:
                    idxes_to_logit[idx] = logits[idx, order]
                idxes_to_logit = sorted(
                    idxes_to_logit.items(), key=lambda x: x[1], reverse=True
                )
                # keep the highest logit as order, set others to next candidate
                for idx, _ in idxes_to_logit[1:]:
                    ret[idx] = orders[idx].pop()
        return ret

    def run_rel_pred(
        self,
        inputs: Dict[str, torch.Tensor],
    ):
        with torch.inference_mode():
            logits = self.model(**inputs).logits.cpu().squeeze(0)
        return logits

    def __call__(
        self,
        bboxes: List[List[int]],
        width: int,
        height: int,
    ) -> List[int]:
        inputs = self.rel_pred_pre(
            bboxes,
            width=width,
            height=height,
        )
        logits = self.run_rel_pred(
            inputs=inputs,
        )
        orders = self.rel_pred_post(
            logits,
            length=len(bboxes),
        )
        return [bboxes[i] for i in orders]
