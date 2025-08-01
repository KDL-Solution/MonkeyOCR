import time
from loguru import logger

from magic_pdf.data.dataset import Dataset
from magic_pdf.libs.clean_memory import clean_memory
from magic_pdf.operators.result import InferenceResult
from magic_pdf.model.batch_analyze_llm import InferenceBatch
from magic_pdf.model.monkeyocr import MonkeyOCR


def doc_analyze(
    dataset: Dataset,
    monkeyocr: MonkeyOCR,
) -> InferenceResult:
    inference_batch = InferenceBatch(
        monkeyocr=monkeyocr,
    )

    model_json = []
    doc_analyze_start = time.time()

    images = []
    for index in range(len(dataset)):
        page_data = dataset.get_page(index)
        img_dict = page_data.get_image()
        images.append(img_dict["img"])

    analyze_result = inference_batch(
        images,
    )

    for index in range(len(dataset)):
        page_data = dataset.get_page(index)
        img_dict = page_data.get_image()
        page_width = img_dict["width"]
        page_height = img_dict["height"]
        result = analyze_result.pop(0)

        page_info = {
            "page_no": index,
            "height": page_height,
            "width": page_width,
        }
        page_dict = {
            "layout_dets": result,
            "page_info": page_info,
        }
        model_json.append(page_dict)

    gc_start = time.time()
    clean_memory(monkeyocr.device)
    gc_time = round(time.time() - gc_start, 2)
    logger.info(f"gc time: {gc_time}")

    doc_analyze_time = round(time.time() - doc_analyze_start, 2)
    doc_analyze_speed = round(len(dataset) / doc_analyze_time, 2)
    logger.info(
        f"doc analyze time: {round(time.time() - doc_analyze_start, 2)},"
        f"speed: {doc_analyze_speed} pages/second"
    )
    return InferenceResult(
        model_json,
        dataset,
    )
