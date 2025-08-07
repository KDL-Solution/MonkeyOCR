import copy
import statistics
import time
import torch
import torch.nn as nn
import hashlib
import copy
import numpy as np
from typing import List, Dict, Any, Tuple
from loguru import logger

from magic_pdf.config.ocr_content_type import BlockType, ContentType
from magic_pdf.data.dataset import BaseDataset
from magic_pdf.data.data_reader_writer import DataWriter
from magic_pdf.libs.bbox import (
    calculate_overlap_area_in_bbox1_area_ratio,
    __is_overlaps_y_exceeds_threshold,
)
from magic_pdf.model.magic_model import MagicModel
from magic_pdf.libs.clean_memory import clean_memory
from magic_pdf.pre_proc.cut import _cut_image_and_table
from magic_pdf.pre_proc.bboxes_detection import _prepare_bboxes_for_layout_split
from magic_pdf.pre_proc.dict_merge import (
    _fill_spans_in_blocks,
    _fix_block_spans,
    _fix_discarded_block,
)
from magic_pdf.pre_proc.span_list_modification import (
    _get_qa_need_list,
    _remove_overlaps_low_confidence_spans,
    _remove_overlaps_min_spans,
)
from magic_pdf.model.monkeyocr import MonkeyOCR
from magic_pdf.model.sub_modules.relation_prediction.xycut import _recursive_xy_cut
from magic_pdf.model.sub_modules.relation_prediction.layoutlmv3 import (
    run_rel_pred,
)


def _compute_md5(
    file_bytes: bytes,
) -> str:
    hasher = hashlib.md5()
    hasher.update(file_bytes)
    return hasher.hexdigest().upper()


def _dict_to_list(input_dict):
    items_list = []
    for _, item in input_dict.items():
        items_list.append(item)
    return items_list


def _construct_page_component(
    blocks,
    layout_bboxes,
    page_id,
    page_w,
    page_h,
    layout_tree,
    images,
    tables,
    interline_equations,
    discarded_blocks,
    need_drop,
    drop_reason,
):
    return {
        "preproc_blocks": blocks,
        "layout_bboxes": layout_bboxes,
        "page_idx": page_id,
        "page_size": [page_w, page_h],
        "_layout_tree": layout_tree,
        "images": images,
        "tables": tables,
        "interline_equations": interline_equations,
        "discarded_blocks": discarded_blocks,
        "need_drop": need_drop,
        "drop_reason": drop_reason,
    }


def _calculate_block_index(fix_blocks, sorted_bboxes):
    if sorted_bboxes is not None:
        for block in fix_blocks:
            line_index_list = []
            if len(block["lines"]) == 0:
                block["index"] = sorted_bboxes.index(block["bbox"])
            else:
                for line in block["lines"]:
                    line["index"] = sorted_bboxes.index(line["bbox"])
                    line_index_list.append(line["index"])
                median_value = statistics.median(line_index_list)
                block["index"] = median_value


            if block["type"] in [
                BlockType.ImageBody,
                BlockType.TableBody,
                BlockType.Title,
                BlockType.InterlineEquation,
            ]:
                if "real_lines" in block:
                    block["virtual_lines"] = copy.deepcopy(block["lines"])
                    block["lines"] = copy.deepcopy(block["real_lines"])
                    del block["real_lines"]
    else:
        block_bboxes = []
        for block in fix_blocks:
            block["bbox"] = [max(0, x) for x in block["bbox"]]
            block_bboxes.append(block["bbox"])

            if block["type"] in [BlockType.ImageBody, BlockType.TableBody]:
                block["virtual_lines"] = copy.deepcopy(block["lines"])
                block["lines"] = copy.deepcopy(block["real_lines"])
                del block["real_lines"]

        random_boxes = np.array(block_bboxes)
        np.random.shuffle(random_boxes)
        res = []
        _recursive_xy_cut(
            np.asarray(random_boxes).astype(int),
            np.arange(len(block_bboxes)),
            res,
        )
        assert len(res) == len(block_bboxes)
        sorted_boxes = random_boxes[np.array(res)].tolist()

        for i, block in enumerate(fix_blocks):
            block["index"] = sorted_boxes.index(block["bbox"])


        sorted_blocks = sorted(fix_blocks, key=lambda b: b["index"])
        line_inedx = 1
        for block in sorted_blocks:
            for line in block["lines"]:
                line["index"] = line_inedx
                line_inedx += 1

    return fix_blocks


def _insert_lines_into_block(
    block_bbox,
    line_height,
    page_w: int,
    page_h: int,
):
    x0, y0, x1, y1 = block_bbox

    block_height = y1 - y0
    block_weight = x1 - x0

    if line_height * 2 < block_height:
        if (
            block_height > page_h * 0.25 and page_w * 0.5 > block_weight > page_w * 0.25
        ):
            lines = int(block_height / line_height) + 1
        else:
            if block_weight > page_w * 0.4:
                lines = 3
                line_height = (y1 - y0) / lines
            elif block_weight > page_w * 0.25:
                lines = int(block_height / line_height) + 1
            else:
                if block_height / block_weight > 1.2:
                    return [[x0, y0, x1, y1]]
                else:
                    lines = 2
                    line_height = (y1 - y0) / lines

        current_y = y0
        lines_positions = []
        for _ in range(lines):
            lines_positions.append(
                [
                    x0,
                    current_y,
                    x1,
                    current_y + line_height,
                ]
            )
            current_y += line_height
        return lines_positions

    else:
        return [[x0, y0, x1, y1]]


def _sort(
    fix_blocks,
    page_w,
    page_h,
    line_height,
    rel_pred: nn.Module,
):
    page_line_list = []

    def add_lines_to_block(b):
        line_bboxes = _insert_lines_into_block(b["bbox"], line_height, page_w, page_h)
        b["lines"] = []
        for line_bbox in line_bboxes:
            b["lines"].append({"bbox": line_bbox, "spans": []})
        page_line_list.extend(line_bboxes)

    for block in fix_blocks:
        if block["type"] in [
            BlockType.Text,
            BlockType.Title,
            BlockType.ImageCaption,
            BlockType.ImageFootnote,
            BlockType.TableCaption,
            BlockType.TableFootnote,
        ]:
            if len(block["lines"]) == 0:
                add_lines_to_block(block)
            elif block["type"] in [BlockType.Title] and len(block["lines"]) == 1 and (block["bbox"][3] - block["bbox"][1]) > line_height * 2:
                block["real_lines"] = copy.deepcopy(block["lines"])
                add_lines_to_block(block)
            else:
                for line in block["lines"]:
                    bbox = line["bbox"]
                    page_line_list.append(bbox)
        elif block["type"] in [
            BlockType.ImageBody,
            BlockType.TableBody,
            BlockType.InterlineEquation,
        ]:
            block["real_lines"] = copy.deepcopy(block["lines"])
            add_lines_to_block(block)

    if len(page_line_list) > 200:
        return None

    x_scale = 1000. / page_w
    y_scale = 1000. / page_h
    boxes = []
    # logger.info(f"Scale: {x_scale}, {y_scale}, Boxes len: {len(page_line_list)}")
    for left, top, right, bottom in page_line_list:
        if left < 0:
            logger.warning(
                f"left < 0, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}"
            )  # noqa: E501
            left = 0
        if right > page_w:
            logger.warning(
                f"right > page_w, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}"
            )  # noqa: E501
            right = page_w
        if top < 0:
            logger.warning(
                f"top < 0, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}"
            )  # noqa: E501
            top = 0
        if bottom > page_h:
            logger.warning(
                f"bottom > page_h, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}"
            )  # noqa: E501
            bottom = page_h

        left = round(left * x_scale)
        top = round(top * y_scale)
        right = round(right * x_scale)
        bottom = round(bottom * y_scale)
        assert (
            1000 >= right >= left >= 0 and 1000 >= bottom >= top >= 0
        ), f"Invalid box. right: {right}, left: {left}, bottom: {bottom}, top: {top}"  # noqa: E126, E121
        boxes.append([left, top, right, bottom])

    with torch.inference_mode():
        orders = run_rel_pred(
            boxes,
            model=rel_pred,
        )
    sorted_bboxes = [page_line_list[i] for i in orders]
    return sorted_bboxes


def _get_line_height(blocks):
    page_line_height_list = []
    for block in blocks:
        if block["type"] in [
            BlockType.Text, BlockType.Title,
            BlockType.ImageCaption, BlockType.ImageFootnote,
            BlockType.TableCaption, BlockType.TableFootnote
        ]:
            for line in block["lines"]:
                bbox = line["bbox"]
                page_line_height_list.append(int(bbox[3] - bbox[1]))
    if len(page_line_height_list) > 0:
        return statistics.median(page_line_height_list)
    else:
        return 10


def _process_groups(
    groups: List[Dict[str, Any]],
    body_key: str,
    caption_key: str,
    footnote_key: str,
) -> Tuple[List, List, List]:
    body_blocks = []
    caption_blocks = []
    footnote_blocks = []
    for i, group in enumerate(groups):
        group[body_key]["group_id"] = i
        body_blocks.append(group[body_key])
        for caption_block in group[caption_key]:
            caption_block["group_id"] = i
            caption_blocks.append(caption_block)
        for footnote_block in group[footnote_key]:
            footnote_block["group_id"] = i
            footnote_blocks.append(footnote_block)
    return body_blocks, caption_blocks, footnote_blocks


def _process_block_list(blocks, body_type, block_type):
    indices = [block["index"] for block in blocks]
    median_index = statistics.median(indices)

    body_bbox = next((block["bbox"] for block in blocks if block.get("type") == body_type), [])

    return {
        "type": block_type,
        "bbox": body_bbox,
        "blocks": blocks,
        "index": median_index,
    }


def _revert_group_blocks(blocks):
    image_groups = {}
    table_groups = {}
    new_blocks = []
    for block in blocks:
        if block["type"] in [BlockType.ImageBody, BlockType.ImageCaption, BlockType.ImageFootnote]:
            group_id = block["group_id"]
            if group_id not in image_groups:
                image_groups[group_id] = []
            image_groups[group_id].append(block)
        elif block["type"] in [BlockType.TableBody, BlockType.TableCaption, BlockType.TableFootnote]:
            group_id = block["group_id"]
            if group_id not in table_groups:
                table_groups[group_id] = []
            table_groups[group_id].append(block)
        else:
            new_blocks.append(block)

    for group_id, blocks in image_groups.items():
        new_blocks.append(_process_block_list(blocks, BlockType.ImageBody, BlockType.Image))

    for group_id, blocks in table_groups.items():
        new_blocks.append(_process_block_list(blocks, BlockType.TableBody, BlockType.Table))

    return new_blocks


def _remove_outside_spans(spans, all_bboxes, all_discarded_blocks):
    def __get_block_bboxes(blocks, block_type_list):
        return [block[0:4] for block in blocks if block[7] in block_type_list]

    image_bboxes = __get_block_bboxes(all_bboxes, [BlockType.ImageBody])
    table_bboxes = __get_block_bboxes(all_bboxes, [BlockType.TableBody])
    other_block_type = []
    for block_type in BlockType.__dict__.values():
        if not isinstance(block_type, str):
            continue
        if block_type not in [BlockType.ImageBody, BlockType.TableBody]:
            other_block_type.append(block_type)
    other_block_bboxes = __get_block_bboxes(all_bboxes, other_block_type)
    discarded_block_bboxes = __get_block_bboxes(all_discarded_blocks, [BlockType.Discarded])

    new_spans = []
    for span in spans:
        span_bbox = span["bbox"]
        span_type = span["type"]

        if any(
            calculate_overlap_area_in_bbox1_area_ratio(
                span_bbox,
                block_bbox
            ) > 0.4 for block_bbox in discarded_block_bboxes
        ):
            new_spans.append(span)
            continue

        if span_type == ContentType.Image:
            if any(
                calculate_overlap_area_in_bbox1_area_ratio(
                    span_bbox,
                    block_bbox,
                ) > 0.5 for block_bbox in image_bboxes
            ):
                new_spans.append(span)
        elif span_type == ContentType.Table:
            if any(
                calculate_overlap_area_in_bbox1_area_ratio(
                    span_bbox,
                    block_bbox,
                ) > 0.5 for block_bbox in table_bboxes
            ):
                new_spans.append(span)
        else:
            if any(
                calculate_overlap_area_in_bbox1_area_ratio(
                    span_bbox,
                    block_bbox
                ) > 0.5 for block_bbox in other_block_bboxes
            ):
                new_spans.append(span)
    return new_spans


def _merge_title_blocks(
    blocks,
    x_distance_thresh,
):
    def __merge_two_bbox(b1, b2):
        x_min = min(b1["bbox"][0], b2["bbox"][0])
        y_min = min(b1["bbox"][1], b2["bbox"][1])
        x_max = max(b1["bbox"][2], b2["bbox"][2])
        y_max = max(b1["bbox"][3], b2["bbox"][3])
        return x_min, y_min, x_max, y_max

    def __merge_two_blocks(b1, b2):
        b1["bbox"] = __merge_two_bbox(b1, b2)
        line1 = b1["lines"][0]
        line2 = b2["lines"][0]
        line1["bbox"] = __merge_two_bbox(line1, line2)
        line1["spans"].extend(line2["spans"])
        return b1, b2

    y_overlapping_blocks = []
    title_bs = [b for b in blocks if b["type"] == BlockType.Title]
    while title_bs:
        block1 = title_bs.pop(0)
        current_row = [block1]
        to_remove = []
        for block2 in title_bs:
            if (
                __is_overlaps_y_exceeds_threshold(block1["bbox"], block2["bbox"], 0.9)
                and len(block1["lines"]) == 1
                and len(block2["lines"]) == 1
            ):
                current_row.append(block2)
                to_remove.append(block2)
        for b in to_remove:
            title_bs.remove(b)
        y_overlapping_blocks.append(current_row)

    to_remove_blocks = []
    for row in y_overlapping_blocks:
        if len(row) == 1:
            continue

        row.sort(key=lambda x: x["bbox"][0])

        merged_block = row[0]
        for i in range(1, len(row)):
            left_block = merged_block
            right_block = row[i]

            left_height = left_block["bbox"][3] - left_block["bbox"][1]
            right_height = right_block["bbox"][3] - right_block["bbox"][1]

            if (
                right_block["bbox"][0] - left_block["bbox"][2] < x_distance_thresh
                and left_height * 0.95 < right_height < left_height * 1.05
            ):
                merged_block, to_remove_block = __merge_two_blocks(merged_block, right_block)
                to_remove_blocks.append(to_remove_block)
            else:
                merged_block = right_block
    for b in to_remove_blocks:
        blocks.remove(b)


def _para_split(pdf_info_dict):
    all_blocks = []
    for page_num, page in pdf_info_dict.items():
        blocks = copy.deepcopy(page['preproc_blocks'])
        for block in blocks:
            block['page_num'] = page_num
            block['page_size'] = page['page_size']
        all_blocks.extend(blocks)

    for page_num, page in pdf_info_dict.items():
        page['para_blocks'] = []
        for block in all_blocks:
            if block['page_num'] == page_num:
                page['para_blocks'].append(block)


def postprocess(
    model_list: List[Dict[str, Any]],
    dataset: BaseDataset,
    image_writer: DataWriter,
    monkeyocr: MonkeyOCR,
    debug_mode=False,
    need_drop = False,  # Fixed.
    drop_reason = [],  # Fixed.
):
    md5 = _compute_md5(dataset.data_bits())

    magic_model = MagicModel(
        model_list=model_list,
        dataset=dataset,
    )

    start_time = time.time()

    pdf_info_dict = {}
    for page_id, fitz_page in enumerate(dataset):
        if debug_mode:
            time_now = time.time()
            logger.info(
                f"page_id: {page_id}, last_page_cost_time: {round(time.time() - start_time, 2)}"
            )
            start_time = time_now

        image_groups = magic_model.get_images(page_id)
        table_groups = magic_model.get_tables(page_id)
        img_body_blocks, img_caption_blocks, img_footnote_blocks = _process_groups(
            image_groups,
            "image_body",
            "image_caption_list",
            "image_footnote_list",
        )
        table_body_blocks, table_caption_blocks, table_footnote_blocks = _process_groups(
            table_groups,
            "table_body",
            "table_caption_list",
            "table_footnote_list",
        )

        discarded_blocks = magic_model.get_discarded(page_id)
        text_blocks = magic_model.get_text_blocks(page_id)
        title_blocks = magic_model.get_title_blocks(page_id)
        _, interline_equations, interline_equation_blocks = magic_model.get_equations(page_id)
        page_w, page_h = magic_model.get_page_size(page_id)

        all_bboxes, all_discarded_blocks = _prepare_bboxes_for_layout_split(
            img_body_blocks,
            img_caption_blocks,
            img_footnote_blocks,
            table_body_blocks,
            table_caption_blocks,
            table_footnote_blocks,
            discarded_blocks,
            text_blocks,
            title_blocks,
            interline_equation_blocks,
            page_w,
            page_h,
        )

        spans = magic_model.get_all_spans(page_id)
        spans = _remove_outside_spans(
            spans,
            all_bboxes,
            all_discarded_blocks,
        )
        spans, _ = _remove_overlaps_low_confidence_spans(spans)
        spans, _ = _remove_overlaps_min_spans(spans)
        discarded_block_with_spans, spans = _fill_spans_in_blocks(
            blocks=all_discarded_blocks,
            spans=spans,
            ratio=0.4
        )
        fix_discarded_blocks = _fix_discarded_block(discarded_block_with_spans)

        if len(all_bboxes) == 0:
            logger.warning(f"skip this page, not found useful bbox, page_id: {page_id}")
            return 
        (
                [],
                [],
                page_id,
                page_w,
                page_h,
                [],
                [],
                [],
                interline_equations,
                fix_discarded_blocks,
                need_drop,
                drop_reason,
            )

        spans = _cut_image_and_table(
            spans,
            fitz_page,
            page_id,
            md5,
            image_writer,
        )
        block_with_spans, spans = _fill_spans_in_blocks(
            blocks=all_bboxes,
            spans=spans,
            ratio=0.5,
        )
        fix_blocks = _fix_block_spans(block_with_spans)

        _merge_title_blocks(
            fix_blocks,
            x_distance_thresh=0.1 * page_w,
        )
        line_height = _get_line_height(fix_blocks)
        _sorted_bboxes = _sort(
            fix_blocks=fix_blocks,
            page_w=page_w,
            page_h=page_h,
            line_height=line_height,
            rel_pred=monkeyocr.rel_pred,
        )  # Relation prediction.
        fix_blocks = _calculate_block_index(
            fix_blocks=fix_blocks,
            sorted_bboxes=_sorted_bboxes,
        )
        fix_blocks = _revert_group_blocks(fix_blocks)
        sorted_blocks = sorted(fix_blocks, key=lambda b: b["index"])

        for block in sorted_blocks:
            if block["type"] in [BlockType.Image, BlockType.Table]:
                block["blocks"] = sorted(block["blocks"], key=lambda b: b["index"])

        images, tables, interline_equations = _get_qa_need_list(sorted_blocks)
        page_info = _construct_page_component(
            sorted_blocks,
            [],
            page_id,
            page_w,
            page_h,
            [],
            images,
            tables,
            interline_equations,
            fix_discarded_blocks,
            need_drop,
            drop_reason,
        )
        pdf_info_dict[f"page_{page_id}"] = page_info

    _para_split(pdf_info_dict)

    pdf_info_list = _dict_to_list(pdf_info_dict)
    new_pdf_info_dict = {
        "pdf_info": pdf_info_list,
    }

    clean_memory(monkeyocr.device)
    return new_pdf_info_dict
