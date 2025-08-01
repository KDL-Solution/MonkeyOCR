class LoRAType:
    BASE = "base"
    TABLE = "table_image_otsl"


class BasePrompts:
    TEXT = """Please output the text content from the image."""
    FORMULA = """Please write out the expression of the formula in the image using LaTeX format."""
    TABLE = """This is the image of a table. Please output the table in html format."""
    Image = """Write a caption describing the image."""


class LoRAPrompts:
    BASE = "Please output the content from the image. Output only the extracted content, no additional text."
    TABLE = "Parse the table in the image."
    TEXT = "Please output the text content from the image. Output only the text content, no additional text."
    Image = """Write a caption describing the image."""
