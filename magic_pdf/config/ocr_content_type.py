class ContentType:
    Image = "image"
    Table = "table"
    Text = "text"
    InlineEquation = "inline_equation"
    InterlineEquation = "interline_equation"


class BlockType:
    Image = "image"
    ImageBody = "image_body"
    ImageCaption = "image_caption"
    ImageFootnote = "image_footnote"
    Table = "table"
    TableBody = "table_body"
    TableCaption = "table_caption"
    TableFootnote = "table_footnote"
    Text = "text"
    Title = "title"
    InterlineEquation = "interline_equation"
    Footnote = "footnote"
    Discarded = "discarded"
    List = "list"
    Index = "index"


class CategoryId:
    Title = 0  # -> `OcrText`.
    Text = 1  # -> `OcrText`.
    Abandon = 2  # -> `OcrText`.
    ImageBody = 3
    ImageCaption = 4  # -> `OcrText`.
    TableBody = 5
    TableCaption = 6  # -> `OcrText`.
    TableFootnote = 7  # -> `OcrText`.
    InterlineEquation_Layout = 8  # -> `InterlineEquation_Layout`.
    # InlineEquation = 13
    InterlineEquation_YOLO = 14  # -> `InterlineEquation_Layout`.
    OcrText = 15
    ImageFootnote = 101  # -> `OcrText`.