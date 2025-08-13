import os
import fitz
import numpy as np
from PIL import Image
from typing import Iterator, Dict
from loguru import logger
from pydantic import BaseModel, Field


class PageInfo(BaseModel):
    w: float = Field(description="the width of page")
    h: float = Field(description="the height of page")


class FitzPage(object):
    def __init__(
        self,
        page: fitz.Page,
    ):
        self.page = page

    def get_image(
        self,
        dpi: int = 200,
    ) -> Dict:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pm = self.page.get_pixmap(matrix=mat, alpha=False)

        # If the width or height exceeds 4500 after scaling, do not scale further.
        if pm.width > 4500 or pm.height > 4500:
            pm = self.page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

        img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
        img = np.array(img)

        img_dict = {"img": img, "width": pm.width, "height": pm.height}
        return img_dict

    def get_page_info(
        self,
    ) -> PageInfo:
        page_w = self.page.rect.width
        page_h = self.page.rect.height
        return PageInfo(w=page_w, h=page_h)


class PDFDataset(object):
    def __init__(
        self,
        bits: bytes,
        lang: str = None,
    ):
        self._raw_fitz = fitz.open("pdf", bits)
        self._pages = [FitzPage(i) for i in self._raw_fitz]
        self._data_bits = bits
        self._raw_data = bits

        if lang == "":
            self._lang = None
        else:
            self._lang = lang
            logger.info(f"Lang: {lang}")

    def __len__(self) -> int:
        return len(self._pages)

    def __iter__(self) -> Iterator[FitzPage]:
        return iter(self._pages)

    def data_bits(self) -> bytes:
        return self._data_bits

    def get_page(
        self,
        page_idx: int,
    ) -> FitzPage:
        return self._pages[page_idx]

    def dump_to_file(self, file_path: str):
        dir_name = os.path.dirname(file_path)
        if dir_name not in ("", ".", ".."):
            os.makedirs(dir_name, exist_ok=True)
        self._raw_fitz.save(file_path)


class DataReader(object):
    def __init__(
        self,
        parent_dir: str = "",
    ):
        self._parent_dir = parent_dir

    def read_at(
        self,
        path: str,
        offset: int = 0,
        limit: int = -1,
    ) -> bytes:
        fn_path = path
        if not os.path.isabs(fn_path) and len(self._parent_dir) > 0:
            fn_path = os.path.join(self._parent_dir, path)

        with open(fn_path, 'rb') as f:
            f.seek(offset)
            if limit == -1:
                return f.read()
            else:
                return f.read(limit)


class DataWriter(object):
    def __init__(
        self,
        parent_dir: str = "",
    ) -> None:
        self._parent_dir = parent_dir

    def write(
        self,
        path: str,
        data: bytes,
    ) -> None:
        fn_path = path
        if not os.path.isabs(fn_path) and len(self._parent_dir) > 0:
            fn_path = os.path.join(self._parent_dir, path)

        if not os.path.exists(os.path.dirname(fn_path)) and os.path.dirname(fn_path) != "":
            os.makedirs(os.path.dirname(fn_path), exist_ok=True)

        with open(fn_path, 'wb') as f:
            f.write(data)
