#!/usr/bin/env python3
# Copyright (c) Opendatalab. All rights reserved.
import os
import time
import argparse
import sys
import logging
import torch.distributed as dist
from typing import List
from pathlib import Path
import fitz

# import sys
# sys.path.insert(0, "/home/eric/workspace/MonkeyOCR/")
# from magic_pdf.config.chat_content_type import TaskInstructions
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PDFDataset, ImageDataset
from magic_pdf.model.doc_analysis import doc_analyze
from magic_pdf.model.monkeyocr import MonkeyOCR
from magic_pdf.operators.result import InferenceResult


def to_pdf_bytes(
    image_bytes_ls: List[bytes],
) -> bytes:
    doc = fitz.open()
    for image_bytes in image_bytes_ls:
        pdf_bytes = fitz.open(stream=image_bytes).convert_to_pdf()
        raw_fitz = fitz.open(
            "pdf",
            pdf_bytes,
        )
        doc.insert_pdf(
            raw_fitz,
        )
    return doc.tobytes()


def parse_folder(
    in_folder: str,
    output_dir: str,
    monkeyocr: MonkeyOCR,
):
    """
    Parse file and save results
    
    Args:
        input_file: Input PDF file path
        output_dir: Output directory
        monkeyocr: Pre-initialized model instance
    """
    # input_file = "/mnt/AI_NAS/Data/경기도청/monkeyocr_test/재해ㆍ재난 위기대응 절차서"
    # input_file = "/mnt/AI_NAS/Data/경기도청/경기도청_샘플데이터/RAG 구축을 위한 공통 자료/20241028_4. 개인정보처리시스템 재해ㆍ재난 위기대응 절차서/20241028_4. 개인정보처리시스템 재해ㆍ재난 위기대응 절차서_page-0016.jpg"
    # output_dir = "/home/eric/workspace/MonkeyOCR/output/"
    print(f"Starting to parse file: {in_folder}")

    # Check if input file exists
    if not os.path.exists(in_folder):
        raise FileNotFoundError(f"Input file does not exist: {in_folder}")

    # Get filename
    # name_without_suff = '.'.join(os.path.basename(input_file).split(".")[:-1])
    name_without_suff = "<|1|>"
    # Prepare output directory
    local_image_dir = os.path.join(output_dir, name_without_suff, "images")
    local_md_dir = os.path.join(output_dir, name_without_suff)
    image_dir = os.path.basename(local_image_dir)
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)

    print(f"Output dir: {local_md_dir}")
    image_writer = FileBasedDataWriter(local_image_dir)
    md_writer = FileBasedDataWriter(local_md_dir)

    image_paths = [
        i.as_posix()
        for i in Path(in_folder).glob("*")
        if "@eaDir" not in i.as_posix()
    ]
    reader = FileBasedDataReader()
    file_bytes = to_pdf_bytes(
        [reader.read_at(i) for i in image_paths],
    )
    dataset = PDFDataset(
        file_bytes,
    )

    # Start inference
    print("Performing document parsing...")
    start_time = time.time()

    infer_result = doc_analyze(
        dataset=dataset,
        monkeyocr=monkeyocr,
    )
    pipe_result = infer_result.pipe_ocr_mode(
        image_writer=image_writer,
        monkeyocr=monkeyocr,
    )
    
    parsing_time = time.time() - start_time
    print(f"Parsing time: {parsing_time:.2f}s")

    infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_draw_model.pdf"))

    pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_draw_layout.pdf"))
    pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_draw_span.pdf"))
    pipe_result.dump_markdown(md_writer, f"{name_without_suff}_dump_markdown.md", image_dir)
    pipe_result.dump_content_list(md_writer, f"{name_without_suff}_dump_content_list.json", image_dir)
    pipe_result.dump_middle_json(md_writer, f'{name_without_suff}_dump_middle.json')

    print("Results saved to ", local_md_dir)
    return local_md_dir


def main():
    parser = argparse.ArgumentParser(
        description="PDF Document Parsing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python parse.py input.pdf                  # Parse single PDF file
  python parse.py input.pdf -o ./output      # Parse single PDF with custom output dir
  python parse.py input.pdf -c model_configs.yaml
  python parse.py image.jpg -t text          # Single task: text recognition
  python parse.py image.jpg -t table         # Single task: table recognition
  python parse.py document.pdf -t text       # Single task: text recognition from all PDF pages (with warning)
  """
)    
    parser.add_argument(
        "input_path",
        help="Input PDF/image file path or folder path"
    )
    parser.add_argument(
        "-o", "--output",
        default="./output",
        help="Output directory (default: ./output)"
    )
    
    parser.add_argument(
        "-c", "--config",
        default="model_configs.yaml",
        help="Configuration file path (default: model_configs.yaml)"
    )
    
    parser.add_argument(
        "-t", "--task",
        choices=['text', 'table'],
        help="Single task recognition type (text/table). Supports both image and PDF files."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    monkeyocr = MonkeyOCR(
        args.config,
    )
    
    # try:
    if os.path.isdir(args.input_path):
        result_dir = parse_folder(
            args.input_path,
            args.output,
            monkeyocr
        )

        if args.task:
            print(f"\n✅ Folder processing with single task ({args.task}) recognition completed! Results saved in: {result_dir}")
        else:
            print(f"\n✅ Folder processing completed! Results saved in: {result_dir}")
    elif os.path.isfile(args.input_path):
        print("Loading model...")

        if args.task:
            result_dir = single_task_recognition(
                args.input_path,
                args.output,
                monkeyocr,
                args.task
            )
            print(f"\n✅ Single task ({args.task}) recognition completed! Results saved in: {result_dir}")
        else:
            result_dir = parse_folder(
                args.input_path,
                args.output,
                monkeyocr
            )
            print(f"\n✅ Parsing completed! Results saved in: {result_dir}")
    else:
        raise FileNotFoundError(f"Input path does not exist: {args.input_path}")

    # except Exception as e:
    #     print(f"\n❌ Processing failed: {str(e)}", file=sys.stderr)
    #     sys.exit(1)
    # finally:
    #     # Clean up resources
    #     try:
    #         if monkeyocr is not None:
    #             # Clean up model resources if needed
    #             if hasattr(monkeyocr, 'chat_model') and hasattr(monkeyocr.chat_model, 'close'):
    #                 monkeyocr.chat_model.close()
                    
    #         # Give time for async tasks to complete before exiting
    #         time.sleep(1.0)
            
    #         if dist.is_initialized():
    #             dist.destroy_process_group()
                
    #     except Exception as cleanup_error:
    #         print(f"Warning: Error during final cleanup: {cleanup_error}")


if __name__ == "__main__":
    main()
