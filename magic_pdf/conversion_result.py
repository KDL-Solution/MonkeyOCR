import copy
import json
import os
from typing import List, Dict, Any
from pathlib import Path

from magic_pdf.data.data_reader_writer import DataWriter
from magic_pdf.data.dataset import BaseDataset
from magic_pdf.pdf_parsing import postprocess
from magic_pdf.model.monkeyocr import MonkeyOCR
from magic_pdf.config.make_content_config import DropMode, MakeMode
from magic_pdf.dict2md.content import union_make
from magic_pdf.libs.draw import (
    draw_model_bbox,
    draw_layout,
    draw_spans,
)


class ConversionResult:
    def __init__(
        self,
        pipe_res,
        dataset: BaseDataset,
    ):
        self._pipe_res = pipe_res
        self._dataset = dataset

    def _get_markdown(
        self,
        drop_mode=DropMode.NONE,
        md_make_mode=MakeMode.MM_MD,
    ) -> str:
        pdf_info_list = self._pipe_res["pdf_info"]
        md_content = union_make(
            pdf_info_list,
            make_mode=md_make_mode,
            drop_mode=drop_mode,
        )
        return md_content.replace("\$", "$").replace("\*", "*").replace("<seg>", "\<seg\>").replace("<sos", "\<sos\>").replace("<eos>", "\<eos\>").replace("<pad>", "\<pad\>").replace("<unk>", "\<unk\>").replace("<sep>", "\<sep\>").replace("<cls>", "\<cls\>")

    def dump_markdown(
        self,
        save_path: str,
        drop_mode=DropMode.NONE,
        md_make_mode=MakeMode.MM_MD,
    ):
        md_out = self._get_markdown(
            drop_mode=drop_mode,
            md_make_mode=md_make_mode,
        )
        Path(save_path).write_text(
            md_out,
            encoding="utf-8",
        )
    def _get_content_list(
        self,
        drop_mode=DropMode.NONE,
    ) -> str:
        pdf_info_list = self._pipe_res["pdf_info"]
        content_list = union_make(
            pdf_info_list,
            make_mode=MakeMode.STANDARD_FORMAT,
            drop_mode=drop_mode,
        )
        return content_list

    def dump_content_list(
        self,
        save_path: str,
        drop_mode=DropMode.NONE,
        indent: int = 2,
    ):
        content_list = self._get_content_list(
            drop_mode=drop_mode,
        )
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


class IntermediateConversionResult:
    def __init__(
        self,
        conv_results: List[Dict[str, Any]],
        dataset: BaseDataset,
    ):
        self.conv_results = conv_results
        self.dataset = dataset

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

    def make_conversion_result(
        self,
        image_writer: DataWriter,
        monkeyocr: MonkeyOCR,
        debug_mode=False,
    ) -> ConversionResult:
        out = postprocess(
            copy.deepcopy(self.conv_results),
            dataset=self.dataset,
            image_writer=image_writer,
            monkeyocr=monkeyocr,
            debug_mode=debug_mode,
        )
        return ConversionResult(
            pipe_res=out,
            dataset=self.dataset,
        )
