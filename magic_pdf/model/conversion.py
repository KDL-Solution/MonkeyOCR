import time
import copy
import json
import os
from loguru import logger
from typing import List, Dict, Any
from pathlib import Path

from magic_pdf.libs.data import PDFDataset, DataWriter
from magic_pdf.libs.clean_memory import clean_memory
from magic_pdf.model.vram import clean_vram
from magic_pdf.model.monkeyocr import MonkeyOCR
from magic_pdf.model.sub_modules.layout_detection.doclayoutyolo import (
    layout_det_pre,
    run_layout_det,
    layout_det_post,
)
from magic_pdf.model.sub_modules.llm.vllm import (
    run_llm,
    llm_post,
)
from magic_pdf.postprocessing import postprocess
from magic_pdf.config import Mode
from magic_pdf.libs.content import union_make
from magic_pdf.libs.draw import (
    draw_model_bbox,
    draw_layout,
    draw_spans,
)


class ConversionResult:
    def __init__(
        self,
        postprocess_out,
        dataset: PDFDataset,
    ):
        self._pipe_res = postprocess_out
        self._dataset = dataset

    def _get_markdown(
        self,
    ) -> str:
        pdf_info_list = self._pipe_res["pdf_info"]
        md_content = union_make(
            pdf_info_list,
            mode=Mode.MARKDOWN,
        )
        return md_content.replace("\$", "$").replace("\*", "*").replace("<seg>", "\<seg\>").replace("<sos", "\<sos\>").replace("<eos>", "\<eos\>").replace("<pad>", "\<pad\>").replace("<unk>", "\<unk\>").replace("<sep>", "\<sep\>").replace("<cls>", "\<cls\>")

    def dump_markdown(
        self,
        save_path: str,
    ):
        md_out = self._get_markdown()
        Path(save_path).write_text(
            md_out,
            encoding="utf-8",
        )
    def _get_content_list(
        self,
    ) -> str:
        pdf_info_list = self._pipe_res["pdf_info"]
        content_list = union_make(
            pdf_info_list,
            mode=Mode.STANDARD,
        )
        return content_list

    def dump_content_list(
        self,
        save_path: str,
        indent: int = 2,
    ):
        content_list = self._get_content_list()
        Path(save_path).write_text(
            json.dumps(
                content_list,
                ensure_ascii=False,
                indent=indent,
            ),
            encoding="utf-8",
        )

    def dump_middle_json(
        self,
        save_path: str,
        indent: int = 2,
    ):
        middle_json = json.dumps(
            self._pipe_res,
            ensure_ascii=False,
            indent=indent,
        )
        Path(save_path).write_text(
            middle_json,
            encoding="utf-8",
        )

    def dump_layout(
        self,
        save_path: str,
    ) -> None:
        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        pdf_info = self._pipe_res["pdf_info"]
        draw_layout(
            pdf_info=pdf_info,
            pdf_bytes=self._dataset.data_bits(),
            save_path=save_path,
        )

    def dump_spans(
        self,
        save_path: str,
    ):
        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        pdf_info = self._pipe_res["pdf_info"]
        draw_spans(
            pdf_info=pdf_info,
            pdf_bytes=self._dataset.data_bits(),
            save_path=save_path,
        )

    def draw_model(
        self,
        save_path: str,
    ) -> None:
        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        draw_model_bbox(
            conv_results=copy.deepcopy(self.conv_results),
            dataset=self.dataset,
            save_path=save_path,
        )

    def dump_model(
        self,
        writer: DataWriter,
        save_path: str,
        indent: int = 2,
    ):
        writer.write_string(
            save_path,
            json.dumps(
                self.conv_results,
                ensure_ascii=False,
                indent=indent,
            )
        )

    def get_infer_res(
        self,
    ):
        return self.conv_results


class Conversion:
    def __init__(
        self,
        image_writer: DataWriter,
        monkeyocr: MonkeyOCR,
    ):
        self.image_writer = image_writer
        self.monkeyocr = monkeyocr

    def make_result(
        self,
        llm_post_out: List[Dict[str, Any]],
        dataset: PDFDataset,
        debug_mode=False,
    ) -> ConversionResult:
        postprocess_out = postprocess(
            copy.deepcopy(llm_post_out),
            dataset=dataset,
            image_writer=self.image_writer,
            monkeyocr=self.monkeyocr,
            debug_mode=debug_mode,
        )
        return ConversionResult(
            postprocess_out=postprocess_out,
            dataset=dataset,
        )

    def __call__(
        self,
        dataset: PDFDataset,
    ):
        ### Layout detection:
        layout_det_start = time.time()

        images = layout_det_pre(
            dataset,
        )  # Same as # pages.
        layout_det_out = run_layout_det(
            images,
            model=self.monkeyocr.layout_det,
        )
        clean_vram(
            self.monkeyocr.device,
            vram_threshold=8,
        )
        layout_det_post_out = layout_det_post(
            images,
            layout_det_out,
        )

        layout_det_time = time.time() - layout_det_start
        layout_det_speed = layout_det_time / len(dataset)
        logger.info(
            f"Layout detection: {round(layout_det_time, 2)}s"
            f" ({round(layout_det_speed, 2)}s/pages)"
        )
        ### : Layout detection

        ### LLM:
        llm_start = time.time()

        llm_out = run_llm(
            layout_det_post_out["images"],
            cat_ids=layout_det_post_out["category_ids"],
            model=self.monkeyocr.llm,
        )  # LLM inference.
        llm_post_out = llm_post(
            dataset=dataset,
            layout_det_out=layout_det_out,
            llm_out=llm_out,
            page_indices=layout_det_post_out["page_indices"],
        )

        llm_time = time.time() - llm_start
        llm_speed = llm_time / len(dataset)
        logger.info(
            f"LLM: {round(llm_time, 2)}s"
            f" ({round(llm_speed, 2)}s/pages)"
        )
        ### : LLM

        gc_start = time.time()
        clean_memory(self.monkeyocr.device)
        gc_time = time.time() - gc_start
        logger.info(f"Garbage collection time: {round(gc_time, 2)}")

        ### Relation prediction, etc.:
        rel_pred_start = time.time()

        final_out = self.make_result(
            llm_post_out=llm_post_out,
            dataset=dataset,
        )

        rel_pred_time = time.time() - rel_pred_start
        rel_pred_speed = rel_pred_time / len(dataset)
        logger.info(
            f"Relation prediction: {round(rel_pred_time, 2)}s"
            f" ({round(rel_pred_speed, 2)}s/pages)"
        )
        ### : Relation prediction, etc.
        return final_out
