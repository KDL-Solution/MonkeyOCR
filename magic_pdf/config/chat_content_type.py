class LoraType:
    BASE = "base"
    TABLE = "table_image_otsl"
    # TEXT = "text"


class TaskInstructions:
    TEXT = """Please output the text content from the image."""
    FORMULA = """Please write out the expression of the formula in the image using LaTeX format."""
    TABLE = """This is the image of a table. Please output the table in html format."""
    Image = """Write a caption describing the image."""


class LoraInstructions:
    BASE = "Please output the content from the image. Output only the extracted content, no additional text."
    # TABLE = "Please output the table content from the image. Output only the table data, no additional text."
    TABLE = """Parse the table in the image."""
    TEXT = "Please output the text content from the image. Output only the text content, no additional text."
    Image = """Write a caption describing the image."""
