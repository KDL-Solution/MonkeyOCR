import copy
from typing import List, Dict, Any, Tuple
from loguru import logger

from magic_pdf.config import BlockType, ContentType
from magic_pdf.libs.bbox import (
    calculate_overlap_area_in_bbox1_area_ratio,
    __is_overlaps_y_exceeds_threshold,
)


def _line_sort_spans_by_left_to_right(
    lines,
):
    line_objects = []
    for line in lines:

        line.sort(key=lambda span: span['bbox'][0])
        line_bbox = [
            min(span['bbox'][0] for span in line),  # x0
            min(span['bbox'][1] for span in line),  # y0
            max(span['bbox'][2] for span in line),  # x1
            max(span['bbox'][3] for span in line),  # y1
        ]
        line_objects.append({
            'bbox': line_bbox,
            'spans': line,
        })
    return line_objects


def _merge_spans_to_line(
    spans,
    threshold: float = 0.6,
):
    if len(spans) == 0:
        return []
    else:

        spans.sort(key=lambda span: span['bbox'][1])

        lines = []
        current_line = [spans[0]]
        for span in spans[1:]:
            if span['type'] in [
                    ContentType.InterlineEquation, ContentType.Image,
                    ContentType.Table
            ] or any(s['type'] in [
                    ContentType.InterlineEquation, ContentType.Image,
                    ContentType.Table
            ] for s in current_line):

                lines.append(current_line)
                current_line = [span]
                continue


            if __is_overlaps_y_exceeds_threshold(span['bbox'], current_line[-1]['bbox'], threshold):
                current_line.append(span)
            else:

                lines.append(current_line)
                current_line = [span]


        if current_line:
            lines.append(current_line)

        return lines


def _fill_spans_in_blocks(
    blocks,
    spans,
    ratio: float,
):
    block_with_spans = []
    for block in blocks:
        block_type = block[7]
        block_bbox = block[0:4]
        block_dict = {
            'type': block_type,
            'bbox': block_bbox,
        }
        if block_type in [
            BlockType.ImageBody, BlockType.ImageCaption, BlockType.ImageFootnote,
            BlockType.TableBody, BlockType.TableCaption, BlockType.TableFootnote
        ]:
            block_dict['group_id'] = block[-1]
        block_spans = []
        for span in spans:
            span_bbox = span['bbox']
            if calculate_overlap_area_in_bbox1_area_ratio(
                    span_bbox,
                    block_bbox
                ) > ratio:
                block_spans.append(span)

        block_dict['spans'] = block_spans
        block_with_spans.append(block_dict)


        if len(block_spans) > 0:
            for span in block_spans:
                spans.remove(span)

    return block_with_spans, spans


def _fix_interline_block(
    block,
):
    block_lines = _merge_spans_to_line(block['spans'])
    sort_block_lines = _line_sort_spans_by_left_to_right(block_lines)
    block['lines'] = sort_block_lines
    del block['spans']
    return block


def _fix_text_block(
    block,
):
    for span in block['spans']:
        if span['type'] == ContentType.InterlineEquation:
            span['type'] = ContentType.InlineEquation
    block_lines = _merge_spans_to_line(block['spans'])
    sort_block_lines = _line_sort_spans_by_left_to_right(block_lines)
    block['lines'] = sort_block_lines
    del block['spans']
    return block


def _fix_block_spans(
    block_with_spans,
):
    fix_blocks = []
    for block in block_with_spans:
        block_type = block['type']

        if block_type in [
            BlockType.Text,
            BlockType.Title,
            BlockType.ImageCaption,
            BlockType.ImageFootnote,
            BlockType.TableCaption,
            BlockType.TableFootnote,
        ]:
            block = _fix_text_block(block)
        elif block_type in [BlockType.InterlineEquation, BlockType.ImageBody, BlockType.TableBody]:
            block = _fix_interline_block(block)
        else:
            continue
        fix_blocks.append(block)
    return fix_blocks


def _fix_discarded_block(
    discarded_block_with_spans,
):
    fix_discarded_blocks = []
    for block in discarded_block_with_spans:
        block = _fix_text_block(block)
        fix_discarded_blocks.append(block)
    return fix_discarded_blocks