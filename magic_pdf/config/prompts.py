import regex
from io import StringIO
from docling_core.types.doc.document import DocTagsDocument, DoclingDocument

from magic_pdf.config.ocr_content_type import CategoryId


def sanitize_md(
    output: str,
):
    cleaned = regex.match(r"<md>.*</md>", output, flags=regex.DOTALL)
    if cleaned is None:
        return output.replace("<md>", "").replace("</md>", "").replace("md\n","").strip()
    return f"""{cleaned[0].replace("<md>", "").replace("</md>", "").strip()}"""


def sanitize_math_formula(
    output: str,
):
    cleaned = regex.match(r"\$\$.*\$\$", output, flags=regex.DOTALL)
    if cleaned is None:
        return output.replace("$$", "").strip()
    return f"""{cleaned[0].replace("$$", "").strip()}"""


def sanitize_html(
    output: str,
):
    otsl_match = regex.search(r"<otsl>.*?</otsl>", output, flags=regex.DOTALL)
    if otsl_match:
        try:
            otsl_text = otsl_match.group(0)
            stream = StringIO(otsl_text)
            table_tag = DocTagsDocument.from_doctags_and_image_pairs(stream, images=None)
            doc = DoclingDocument.load_from_doctags(table_tag)
            table_html = []
            for table in doc.tables:
                table_html.append(table.export_to_html(doc=doc))
            if len(table_html) == 0:
                return otsl_text.replace("<otsl>", "").replace("</otsl>", "").strip()
            # If there are tables, return the first table"s HTML
            table_html = "\n".join(table_html)
            return table_html.replace("<otsl>", "").replace("</otsl>", "").strip()
        except Exception as e:
            return otsl_text.replace("<otsl>", "").replace("</otsl>", "").strip()
    
    cleaned = regex.match(r"```html.*```", output, flags=regex.DOTALL)
    if cleaned is None:
        return "<html>\n"+output.replace("```html","<html>").replace("```","</html>").strip()+"\n</html>"
    return f"""{cleaned[0].replace("```html","<html>").replace("```","</html>").strip()}"""


class ModelNames:
    TEXT = "Qwen2.5-VL-7B-Instruct"
    FORMULA = "Qwen2.5-VL-7B-Instruct"
    IMAGE = "Qwen2.5-VL-7B-Instruct"
    TABLE = "table_image_otsl"


class Prompts:
    TEXT = "Please output the text content from the image."
    FORMULA = "Please write out the expression of the formula in the image using LaTeX format."
    IMAGE = "Write a caption describing the image."
    TABLE = "Parse the table in the image."


class PromptConfig:
    CATEGORY_MAPPING = {
        CategoryId.Title: {
            "model_name": ModelNames.TEXT,
            "prompt": Prompts.TEXT,
            "sanitizer": sanitize_md
        },
        CategoryId.Text: {
            "model_name": ModelNames.TEXT,
            "prompt": Prompts.TEXT,
            "sanitizer": sanitize_md
        },
        CategoryId.Abandon: {
            "model_name": ModelNames.TEXT,
            "prompt": Prompts.TEXT,
            "sanitizer": sanitize_md
        },
        CategoryId.ImageBody: {
            "model_name": ModelNames.IMAGE,
            "prompt": Prompts.IMAGE,
            "sanitizer": sanitize_md
        },
        CategoryId.ImageCaption: {
            "model_name": ModelNames.TEXT,
            "prompt": Prompts.TEXT,
            "sanitizer": sanitize_md
        },
        CategoryId.TableBody: {
            "model_name": ModelNames.TABLE,
            "prompt": Prompts.TABLE,
            "sanitizer": sanitize_html
        },
        CategoryId.TableCaption: {
            "model_name": ModelNames.TEXT,
            "prompt": Prompts.TEXT,
            "sanitizer": sanitize_md
        },
        CategoryId.TableFootnote: {
            "model_name": ModelNames.TEXT,
            "prompt": Prompts.TEXT,
            "sanitizer": sanitize_md
        },
        CategoryId.InterlineEquation_Layout: {
            "model_name": ModelNames.TEXT,
            "prompt": Prompts.TEXT,
            "sanitizer": sanitize_math_formula
        },
        CategoryId.InterlineEquation_YOLO: {
            "model_name": ModelNames.TEXT,
            "prompt": Prompts.TEXT,
            "sanitizer": sanitize_math_formula
        },
        CategoryId.ImageFootnote: {
            "model_name": ModelNames.TEXT,
            "prompt": Prompts.TEXT,
            "sanitizer": sanitize_md
        },
    }

    @classmethod
    def get_user_prompt(
        cls,
        category_id,
    ):
        mapping = cls.CATEGORY_MAPPING.get(category_id, {})
        return mapping.get("prompt")

    @classmethod
    def get_model_name(
        cls,
        category_id,
    ):
        mapping = cls.CATEGORY_MAPPING.get(category_id, {})
        return mapping.get("model_name")

    @classmethod
    def get_sanitizer(
        cls,
        category_id,
    ):
        """CategoryId에 해당하는 sanitizer 타입 반환"""
        return cls.CATEGORY_MAPPING.get(category_id, {}).get("sanitizer", sanitize_md)

    @classmethod
    def is_supported_category(
        cls,
        category_id,
    ):
        """지원되는 CategoryId인지 확인"""
        return category_id in cls.CATEGORY_MAPPING
