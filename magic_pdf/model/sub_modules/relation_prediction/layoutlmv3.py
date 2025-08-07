import torch
from collections import defaultdict
from typing import List, Dict
from transformers import LayoutLMv3ForTokenClassification


def rel_pred_pre(
    boxes: List[List[int]],
    model: LayoutLMv3ForTokenClassification,
    cls_token_id = 0,
    unk_token_id = 3,
    eos_token_id = 2,
) -> Dict[str, torch.Tensor]:
    bbox = [[0, 0, 0, 0]] + boxes + [[0, 0, 0, 0]]
    input_ids = [cls_token_id] + [unk_token_id] * len(boxes) + [eos_token_id]
    attention_mask = [1] + [1] * len(boxes) + [1]
    inputs = {
        "bbox": torch.tensor([bbox]),
        "attention_mask": torch.tensor([attention_mask]),
        "input_ids": torch.tensor([input_ids]),
    }
    new_inputs = {}
    for k, v in inputs.items():
        v = v.to(model.device)
        if torch.is_floating_point(v):
            v = v.to(model.dtype)
        new_inputs[k] = v
    return new_inputs


def rel_pred_post(
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
    boxes: List[List[int]],
    model,
) -> List[int]:
    inputs = rel_pred_pre(
        boxes,
        model=model,
    )
    logits = model(**inputs).logits.cpu().squeeze(0)
    return rel_pred_post(logits, len(boxes))
