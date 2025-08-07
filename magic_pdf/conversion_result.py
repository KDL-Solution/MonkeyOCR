import copy
import json
import os
from typing import List, Dict, Any

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
        """Initialized.

        Args:
            pipe_res (list[dict]): the pipeline processed result of model inference result
            dataset (Dataset): the dataset associated with pipe_res
        """
        self._pipe_res = pipe_res
        self._dataset = dataset

    def _get_markdown(
        self,
        img_dir_or_bucket_prefix: str,
        drop_mode=DropMode.NONE,
        md_make_mode=MakeMode.MM_MD,
    ) -> str:
        """Get markdown content.

        Args:
            img_dir_or_bucket_prefix (str): The s3 bucket prefix or local file directory which used to store the figure
            drop_mode (str, optional): Drop strategy when some page which is corrupted or inappropriate. Defaults to DropMode.NONE.
            md_make_mode (str, optional): The content Type of Markdown be made. Defaults to MakeMode.MM_MD.

        Returns:
            str: return markdown content
        """
        pdf_info_list = self._pipe_res["pdf_info"]
        md_content = union_make(
            pdf_info_list,
            make_mode=md_make_mode,
            drop_mode=drop_mode,
            img_buket_path=img_dir_or_bucket_prefix,
        )
        return md_content.replace("\$", "$").replace("\*", "*").replace("<seg>", "\<seg\>").replace("<sos", "\<sos\>").replace("<eos>", "\<eos\>").replace("<pad>", "\<pad\>").replace("<unk>", "\<unk\>").replace("<sep>", "\<sep\>").replace("<cls>", "\<cls\>")

    def dump_markdown(
        self,
        writer: DataWriter,
        file_path: str,
        img_dir_or_bucket_prefix: str,
        drop_mode=DropMode.NONE,
        md_make_mode=MakeMode.MM_MD,
    ):
        """Dump The Markdown.

        Args:
            writer (DataWriter): File writer handle
            file_path (str): The file location of markdown
            img_dir_or_bucket_prefix (str): The s3 bucket prefix or local file directory which used to store the figure
            drop_mode (str, optional): Drop strategy when some page which is corrupted or inappropriate. Defaults to DropMode.NONE.
            md_make_mode (str, optional): The content Type of Markdown be made. Defaults to MakeMode.MM_MD.
        """

        md_content = self._get_markdown(
            img_dir_or_bucket_prefix,
            drop_mode=drop_mode,
            md_make_mode=md_make_mode,
        )
        writer.write_string(file_path, md_content)

    def _get_content_list(
        self,
        image_dir_or_bucket_prefix: str,
        drop_mode=DropMode.NONE,
    ) -> str:
        """Get Content List.

        Args:
            image_dir_or_bucket_prefix (str): The s3 bucket prefix or local file directory which used to store the figure
            drop_mode (str, optional): Drop strategy when some page which is corrupted or inappropriate. Defaults to DropMode.NONE.

        Returns:
            str: content list content
        """
        pdf_info_list = self._pipe_res["pdf_info"]
        content_list = union_make(
            pdf_info_list,
            make_mode=MakeMode.STANDARD_FORMAT,
            drop_mode=drop_mode,
            img_buket_path=image_dir_or_bucket_prefix,
        )
        return content_list

    def dump_content_list(
        self,
        writer: DataWriter,
        file_path: str,
        image_dir_or_bucket_prefix: str,
        drop_mode=DropMode.NONE,
    ):
        """Dump Content List.

        Args:
            writer (DataWriter): File writer handle
            file_path (str): The file location of content list
            image_dir_or_bucket_prefix (str): The s3 bucket prefix or local file directory which used to store the figure
            drop_mode (str, optional): Drop strategy when some page which is corrupted or inappropriate. Defaults to DropMode.NONE.
        """
        content_list = self._get_content_list(
            image_dir_or_bucket_prefix,
            drop_mode=drop_mode,
        )
        writer.write_string(
            file_path, json.dumps(content_list, ensure_ascii=False, indent=4)
        )

    def _get_middle_json(self) -> str:
        """Get middle json.

        Returns:
            str: The content of middle json
        """
        return json.dumps(
            self._pipe_res,
            ensure_ascii=False,
            indent=4,
        )

    def dump_middle_json(self, writer: DataWriter, file_path: str):
        """Dump the result of pipeline.

        Args:
            writer (DataWriter): File writer handler
            file_path (str): The file location of middle json
        """
        middle_json = self._get_middle_json()
        writer.write_string(file_path, middle_json)

    def draw_layout(
        self,
        file_path: str,
    ) -> None:
        """Draw the layout.

        Args:
            file_path (str): The file location of layout result file
        """
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        pdf_info = self._pipe_res["pdf_info"]
        draw_layout(
            pdf_info=pdf_info,
            dataset=self._dataset.data_bits(),
            save_dir=dir_name,
            filename=base_name,
        )

    def draw_span(self, file_path: str):
        """Draw the Span.

        Args:
            file_path (str): The file location of span result file
        """
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        pdf_info = self._pipe_res["pdf_info"]
        draw_spans(
            pdf_info=pdf_info,
            dataset=self._dataset.data_bits(),
            save_dir=dir_name,
            filename=base_name,
        )


class IntermediateConversionResult:
    def __init__(
        self,
        conv_results: List[Dict[str, Any]],
        dataset: BaseDataset,
    ):
        """Initialized method.

        Args:
            inference_results (list): the inference result generated by model
            dataset (Dataset): the dataset related with model inference result
        """
        self.conv_results = conv_results
        self.dataset = dataset

    def draw_model(
        self,
        file_path: str,
    ) -> None:
        """Draw model inference result.

        Args:
            file_path (str): the output file path
        """
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        draw_model_bbox(
            conv_results=copy.deepcopy(self.conv_results),
            dataset=self.dataset,
            save_dir=dir_name,
            filename=base_name,
        )

    def dump_model(
        self,
        writer: DataWriter,
        file_path: str,
    ):
        """Dump model inference result to file.

        Args:
            writer (DataWriter): writer handle
            file_path (str): the location of target file
        """
        writer.write_string(
            file_path, json.dumps(self.conv_results, ensure_ascii=False, indent=4)
        )

    def get_infer_res(
        self,
    ):
        """Get the inference result.

        Returns:
            list: the inference result generated by model
        """
        return self.conv_results

    def make_conversion_result(
        self,
        image_writer: DataWriter,
        monkeyocr: MonkeyOCR,
        debug_mode=False,
    ) -> ConversionResult:
        """Post-proc the model inference result, Extract the text using `OCR`
        technical.

        Args:
            imageWriter (DataWriter): the image writer handle
            start_page_id (int, optional): Defaults to 0. Let user select some pages He/She want to process
            end_page_id (int, optional):  Defaults to the last page index of dataset. Let user select some pages He/She want to process
            debug_mode (bool, optional): Defaults to False. will dump more log if enabled
            lang (str, optional): Defaults to None.

        Returns:
            PipeResult: the result
        """

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
