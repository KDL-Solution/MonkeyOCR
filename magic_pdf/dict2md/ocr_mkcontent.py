import re
from loguru import logger

from magic_pdf.config.make_content_config import DropMode, MakeMode
from magic_pdf.config.ocr_content_type import BlockType, ContentType
from magic_pdf.libs.commons import join_path
from magic_pdf.libs.language import detect_lang
from magic_pdf.post_proc.para_split_v3 import ListLineTag


def _ocr_escape_special_markdown_char(content):
    special_chars = ["*", "`", "~", "$"]
    for char in special_chars:
        content = content.replace(char, "\\" + char)
    return content


def __is_hyphen_at_line_end(line):
    """Check if a line ends with one or more letters followed by a hyphen.

    Args:
    line (str): The line of text to check.

    Returns:
    bool: True if the line ends with one or more letters followed by a hyphen, False otherwise.
    """
    # Use regex to check if the line ends with one or more letters followed by a hyphen
    return bool(re.search(r"[A-Za-z]+-\s*$", line))


def _get_title_level(block):
    title_level = block.get("level", 1)
    if title_level > 4:
        title_level = 4
    elif title_level < 1:
        title_level = 1
    return title_level


def _ocr_mk_markdown_with_para_core(
    paras_of_layout,
    mode,
    img_buket_path="",
):
    page_markdown = []
    for para_block in paras_of_layout:
        para_text = ""
        para_type = para_block["type"]
        if para_type in [
            BlockType.Text,
            BlockType.List,
            BlockType.Index,
        ]:
            para_text = _merge_para_with_text(para_block)
        elif para_type == BlockType.Title:
            title_level = _get_title_level(para_block)
            para_text = f"""{"#" * title_level} {_merge_para_with_text(para_block)}"""
        elif para_type == BlockType.InterlineEquation:
            para_text = _merge_para_with_text(para_block)
        elif para_type == BlockType.Image:
            if mode == MakeMode.NLP_MD:
                continue
            elif mode == MakeMode.MM_MD:
                for block in para_block["blocks"]:
                    if block["type"] == BlockType.ImageBody:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if span["type"] == ContentType.Image:
                                    if span.get("image_path", ""):
                                        para_text += f"""\n![]({join_path(img_buket_path, span["image_path"])})  \n"""
                for block in para_block["blocks"]:
                    if block["type"] == BlockType.ImageCaption:
                        para_text += _merge_para_with_text(block) + "  \n"
                for block in para_block["blocks"]:
                    if block["type"] == BlockType.ImageFootnote:
                        para_text += _merge_para_with_text(block) + "  \n"
        elif para_type == BlockType.Table:
            if mode == MakeMode.NLP_MD:
                continue
            elif mode == MakeMode.MM_MD:
                for block in para_block["blocks"]:
                    if block["type"] == BlockType.TableCaption:
                        para_text += _merge_para_with_text(block) + "  \n"
                for block in para_block["blocks"]:
                    if block["type"] == BlockType.TableBody:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if span["type"] == ContentType.Table:
                                    # if processed by table model
                                    if span.get("latex", ""):
                                        para_text += f"""\n\n$\n {span["latex"]}\n$\n\n"""
                                    elif span.get("html", ""):
                                        para_text += f"\n\n{span['html']}\n\n"
                                    elif span.get("image_path", ""):
                                        para_text += f"\n![]({join_path(img_buket_path, span['image_path'])})  \n"
                for block in para_block["blocks"]:
                    if block["type"] == BlockType.TableFootnote:
                        para_text += _merge_para_with_text(block) + "  \n"

        if para_text.strip() == "":
            continue
        else:
            page_markdown.append(para_text.strip() + "  ")

    return page_markdown


def _merge_para_with_text(para_block):
    block_text = ""
    for line in para_block["lines"]:
        for span in line["spans"]:
            if span["type"] in [ContentType.Text]:
                block_text += span["content"]
    block_lang = detect_lang(block_text[:100])

    para_text = ""
    for i, line in enumerate(para_block["lines"]):

        if i >= 1 and line.get(ListLineTag.IS_LIST_START_LINE, False):
            para_text += "  \n"

        for j, span in enumerate(line["spans"]):

            span_type = span["type"]
            content = ""
            if span_type == ContentType.Text:
                content = _ocr_escape_special_markdown_char(span["content"])
            elif span_type == ContentType.InlineEquation:
                content = f"${span['content']}$"
            elif span_type == ContentType.InterlineEquation:
                content = f"\n$$\n{span['content']}\n$$\n"

            content = content.strip()

            if content:
                langs = ["zh", "ja", "ko"]
                # logger.info(f"block_lang: {block_lang}, content: {content}")
                if block_lang in langs: # In Chinese/Japanese/Korean context, line breaks don"t need space separation, but if it"s inline equation ending, still need to add space
                    if j == len(line["spans"]) - 1 and span_type not in [ContentType.InlineEquation]:
                        para_text += content
                    else:
                        para_text += f"{content} "
                else:
                    if span_type in [ContentType.Text, ContentType.InlineEquation]:
                        # If span is last in line and ends with hyphen, no space should be added at end, and hyphen should be removed
                        if j == len(line["spans"])-1 and span_type == ContentType.Text and __is_hyphen_at_line_end(content):
                            para_text += content[:-1]
                        else:  # In Western text context, content needs space separation
                            para_text += f"{content} "
                    elif span_type == ContentType.InterlineEquation:
                        para_text += content
            else:
                continue
    # Split connected characters
    # para_text = __replace_ligatures(para_text)
    return para_text


def _para_to_standard_format(
    para_block,
    img_buket_path,
    page_idx,
    drop_reason=None,
):
    para_type = para_block["type"]
    para_content = {}
    if para_type in [BlockType.Text, BlockType.List, BlockType.Index]:
        para_content = {
            "type": "text",
            "text": _merge_para_with_text(para_block),
        }
    elif para_type == BlockType.Title:
        title_level = _get_title_level(para_block)
        para_content = {
            "type": "text",
            "text": _merge_para_with_text(para_block),
            "text_level": title_level,
        }
    elif para_type == BlockType.InterlineEquation:
        para_content = {
            "type": "equation",
            "text": _merge_para_with_text(para_block),
            "text_format": "latex",
        }
    elif para_type == BlockType.Image:
        para_content = {"type": "image", "img_path": "", "img_caption": [], "img_footnote": []}
        for block in para_block["blocks"]:
            if block["type"] == BlockType.ImageBody:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["type"] == ContentType.Image:
                            if span.get("image_path", ""):
                                para_content["img_path"] = join_path(img_buket_path, span["image_path"])
            if block["type"] == BlockType.ImageCaption:
                para_content["img_caption"].append(_merge_para_with_text(block))
            if block["type"] == BlockType.ImageFootnote:
                para_content["img_footnote"].append(_merge_para_with_text(block))
    elif para_type == BlockType.Table:
        para_content = {
            "type": "table",
            "img_path": "",
            "table_caption": [],
            "table_footnote": [],
        }
        for block in para_block["blocks"]:
            if block["type"] == BlockType.TableBody:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["type"] == ContentType.Table:

                            if span.get("latex", ""):
                                para_content["table_body"] = f"\n\n$\n {span['latex']}\n$\n\n"
                            elif span.get("html", ""):
                                para_content["table_body"] = f"\n\n{span['html']}\n\n"

                            if span.get("image_path", ""):
                                para_content["img_path"] = join_path(img_buket_path, span["image_path"])

            if block["type"] == BlockType.TableCaption:
                para_content["table_caption"].append(_merge_para_with_text(block))
            if block["type"] == BlockType.TableFootnote:
                para_content["table_footnote"].append(_merge_para_with_text(block))

    para_content["page_idx"] = page_idx

    if drop_reason is not None:
        para_content["drop_reason"] = drop_reason

    return para_content


def union_make(
    pdf_info_dict: list,
    make_mode: str,
    drop_mode: str,
    img_buket_path: str = "",
):
    output_content = []
    for page_info in pdf_info_dict:
        drop_reason = None
        if page_info.get("need_drop", False):
            drop_reason = page_info.get("drop_reason")
            if drop_mode == DropMode.NONE:
                pass
            elif drop_mode == DropMode.WHOLE_PDF:
                raise Exception((f"drop_mode is {DropMode.WHOLE_PDF} ,"
                                 f"drop_reason is {drop_reason}"))
            elif drop_mode == DropMode.SINGLE_PAGE:
                logger.warning((f"drop_mode is {DropMode.SINGLE_PAGE} ,"
                                f"drop_reason is {drop_reason}"))
                continue
            else:
                raise Exception("drop_mode can not be null")

        paras_of_layout = page_info.get("para_blocks")
        page_idx = page_info.get("page_idx")
        if not paras_of_layout:
            continue
        if make_mode == MakeMode.MM_MD:
            page_markdown = _ocr_mk_markdown_with_para_core(
                paras_of_layout,
                mode=MakeMode.MM_MD,
                img_buket_path=img_buket_path,
            )
            output_content.extend(page_markdown)
        elif make_mode == MakeMode.NLP_MD:
            page_markdown = _ocr_mk_markdown_with_para_core(
                paras_of_layout,
                mode=MakeMode.NLP_MD,
            )
            output_content.extend(page_markdown)
        elif make_mode == MakeMode.STANDARD_FORMAT:
            for para_block in paras_of_layout:
                para_content = _para_to_standard_format(
                    para_block,
                    img_buket_path,
                    page_idx,
                )
                output_content.append(para_content)

    if make_mode in [
        MakeMode.MM_MD,
        MakeMode.NLP_MD,
    ]:
        return "\n\n".join(output_content)  # `str`.
    elif make_mode == MakeMode.STANDARD_FORMAT:
        return output_content
