import time
import sys
import logging
import fitz
from typing import List
from pathlib import Path

from magic_pdf.libs.data import (
    PDFDataset,
    DataWriter,
    DataReader,
)
from magic_pdf.model.conversion import Conversion
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


def convert(
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

    image_writer = DataWriter(
        parent_dir=images_dir.as_posix(),
    )
    conv = Conversion(
        image_writer=image_writer,
        monkeyocr=monkeyocr,
    )
    conv_result = conv(
        dataset,
    )

    conv_time = time.time() - conv_start
    print(f"Document conversion time: {conv_time:.2f}s")

    conv_result.dump_markdown(
        save_path=(save_dir / "document_conversion.md").as_posix(),
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


def convert_folder(
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
    reader = DataReader()
    convert(
        [reader.read_at(i) for i in image_paths],
        save_dir=save_dir.as_posix(),
        monkeyocr=monkeyocr,
    )
    return save_dir.as_posix()


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


def convert_pdf(
    pdf_path: str,
    save_dir: str,
    monkeyocr: MonkeyOCR,
):
    save_dir = Path(save_dir) / Path(pdf_path).stem

    image_bytes_ls = _pdf_to_image_bytes_ls(
        pdf_path=pdf_path,
    )
    convert(
        image_bytes_ls,
        save_dir=save_dir.as_posix(),
        monkeyocr=monkeyocr,
    )
    return save_dir.as_posix()


def main(
    in_path: str,
    save_dir: str = "./output",
    config_path: str = "./model_configs.yaml", 
    log_level: str = "INFO",
) -> None:
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    monkeyocr = MonkeyOCR(
        config_path,
    )
    # try:
    in_path = Path(in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {in_path.as_posix()}")

    if in_path.is_dir():
        result_dir = convert_folder(
            in_path.as_posix(),
            save_dir,
            monkeyocr,
        )
    elif in_path.is_file():
        if in_path.suffix == ".pdf":
            result_dir = convert_pdf(
                pdf_path=in_path.as_posix(),
                save_dir=save_dir,
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
    from fire import Fire

    Fire(
        main,
    )
