import os
import time
import argparse
import sys
import logging
import fitz
from typing import List
from pathlib import Path

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PDFDataset
from magic_pdf.model.conversion import convert
from magic_pdf.model.monkeyocr import MonkeyOCR


def _to_pdf_bytes(
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


def _pdf_to_image_bytes_ls(
    pdf_path: str,
) -> List[bytes]:
    doc = fitz.open(pdf_path)
    image_bytes_list = []

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=200)  # 해상도 조절 가능

        # Pixmap → PNG bytes
        image_bytes = pix.tobytes("png")  # "ppm"도 가능
        image_bytes_list.append(image_bytes)
    return image_bytes_list


def document_convert_folder(
    in_folder: str,
    save_dir: str,
    monkeyocr: MonkeyOCR,
):
    in_folder = Path(in_folder)
    save_dir = Path(save_dir) / in_folder.stem

    image_paths = [
        i.as_posix()
        for i in in_folder.glob("*")
        if "@eaDir" not in i.as_posix()
    ]
    reader = FileBasedDataReader()
    document_convert(
        [reader.read_at(i) for i in image_paths],
        save_dir=save_dir.as_posix(),
        monkeyocr=monkeyocr,
    )
    return save_dir.as_posix()


def document_convert_pdf(
    pdf_path: str,
    save_dir: str,
    monkeyocr: MonkeyOCR,
):
    save_dir = Path(save_dir) / Path(pdf_path).stem

    image_bytes_ls = _pdf_to_image_bytes_ls(
        pdf_path=pdf_path,
    )
    document_convert(
        image_bytes_ls,
        save_dir=save_dir.as_posix(),
        monkeyocr=monkeyocr,
    )
    return save_dir.as_posix()


def document_convert(
    image_bytes_ls: List[bytes],
    save_dir: str,
    monkeyocr: MonkeyOCR,
    debug_mode: bool = True,
):
    save_dir = Path(save_dir)
    images_dir = save_dir / "images"
    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_bytes = _to_pdf_bytes(
        image_bytes_ls,
    )
    dataset = PDFDataset(
        pdf_bytes,
    )

    print("Performing document conversion...")
    conv_start = time.time()

    image_writer = FileBasedDataWriter(
        parent_dir=images_dir.as_posix(),
    )
    conv_result = convert(
        dataset=dataset,
        image_writer=image_writer,
        monkeyocr=monkeyocr,
    )

    conv_time = time.time() - conv_start
    print(f"Document conversion time: {conv_time:.2f}s")

    conv_result.dump_markdown(
        save_path="document_conversion.md",
    )  # 우리가 원하는 것.

    if debug_mode:
        # conv_result.draw_model(os.path.join(local_md_dir, f"draw_model.pdf"))
        conv_result.dump_layout(
            save_path=(save_dir / "layout.pdf").as_posix(),
        )
        conv_result.dump_spans(
            save_path=(save_dir / "spans.pdf").as_posix(),
        )
        conv_result.dump_content_list(
            save_path=(save_dir / "content_list.json").as_posix(),
        )
        conv_result.dump_middle_json(
            save_path=(save_dir / "middle.json").as_posix(),
        )


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
        "--save_dir",
        default="./output",
        help="Output directory (default: ./output)"
    )
    
    parser.add_argument(
        "-c", "--config",
        default="model_configs.yaml",
        help="Configuration file path (default: model_configs.yaml)"
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
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file does not exist: {args.input_path}")

    if os.path.isdir(args.input_path):
        result_dir = document_convert_folder(
            args.input_path,
            args.save_dir,
            monkeyocr,
        )
    elif os.path.isfile(args.input_path):
        if Path(args.input_path).suffix == ".pdf":
            result_dir = document_convert_pdf(
                pdf_path=args.input_path,
                save_dir=args.save_dir,
                monkeyocr=monkeyocr,
            )
    print(f"\n✅ Parsing completed! Results saved in: {result_dir}")

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
