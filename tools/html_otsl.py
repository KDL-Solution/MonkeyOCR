from io import StringIO
from docling_core.types.doc.document import DocTagsDocument, DoclingDocument


#otsl_text = "<otsl><fcel>Method<fcel>Size<fcel>Edit Distance ↓<fcel>F1-score ↑<fcel>Precision ↑<fcel>Recall ↑<fcel>BLEU ↑<fcel>METEOR ↑<nl><fcel>Full-page<lcel><lcel><lcel><lcel><lcel><lcel><lcel><nl><fcel>Qwen2.5 VL [9]<fcel>7B<fcel>0.56<fcel>0.72<fcel>0.80<fcel>0.70<fcel>0.46<fcel>0.57<nl><fcel>GOT [89]<fcel>580M<fcel>0.61<fcel>0.69<fcel>0.71<fcel>0.73<fcel>0.48<fcel>0.59<nl><fcel>Nougat (base) [12]<fcel>350M<fcel>0.62<fcel>0.66<fcel>0.72<fcel>0.67<fcel>0.44<fcel>0.54<nl><fcel>SmolDocling (Ours)<fcel>256M<fcel>0.48<fcel>0.80<fcel>0.89<fcel>0.79<fcel>0.58<fcel>0.67<nl><fcel>Code listings<lcel><lcel><lcel><lcel><lcel><lcel><lcel><nl><fcel>SmolDocling (Ours)<fcel>256M<fcel>0.11<fcel>0.92<fcel>0.94<fcel>0.91<fcel>0.87<fcel>0.89<nl><fcel>Equations<lcel><lcel><lcel><lcel><lcel><lcel><lcel><nl><fcel>Qwen2.5 VL [9]<fcel>7B<fcel>0.22<fcel>0.89<fcel>0.91<fcel>0.87<fcel>0.68<fcel>0.77<nl><fcel>GOT [89]<fcel>580M<fcel>0.11<fcel>0.95<fcel>0.95<fcel>0.96<fcel>0.85<fcel>0.91<nl><fcel>Nougat (base) [12]<fcel>350M<fcel>0.62<fcel>0.60<fcel>0.60<fcel>0.53<fcel>0.33<fcel>0.41<nl><fcel>SmolDocling (Ours)<fcel>256M<fcel>0.11<fcel>0.95<fcel>0.96<fcel>0.95<fcel>0.83<fcel>0.89</otsl>"
otsl_text = "<otsl><fcel>요구사항분류<fcel>고유번호<fcel>요구사항정의<nl><fcel>기능요구사항<fcel>공통사항<fcel>SFR-01[공통]개발 공통 준수사항<nl><ucel><fcel>데이터웨어하우스소프트웨어<fcel>DW-SFR-01[데이터웨어하우스]데이터 수집·적재 관리 기능<nl><ucel><ucel><fcel>DW-SFR-02[데이터웨어하우스]헬스케어 데이터 수집 범위<nl><ucel><ucel><fcel>DW-SFR-03[데이터웨어하우스]데이터 표준화 관리 기능<nl><ucel><ucel><fcel>DW-SFR-04[데이터웨어하우스]데이터 전처리 기능<nl><ucel><ucel><fcel>DW-SFR-05[데이터웨어하우스]AI모델 구축·활용을 위한 AI 프레임워크 기능<nl><ucel><ucel><fcel>DW-SFR-06[데이터웨어하우스]거버넌스 및 관리 기능<nl><ucel><fcel>리빙램플랫폼<fcel>LP-SFR-01[리빙랩]데이터웨어하우스 데이터 연동 및 AI 프레임워크 기능 연계<nl><ucel><ucel><fcel>LP-SFR-02[리빙랩]리빙랩 소개 기능<nl><ucel><ucel><fcel>LP-SFR-03[리빙랩]헬스케어 데이터 활용 기능<nl><ucel><ucel><fcel>LP-SFR-04[리빙랩]회원 관리 기능<nl><ucel><ucel><fcel>LP-SFR-05[리빙랩]공공데이터 조회 및 다운로드 기능<nl><ucel><ucel><fcel>LP-SFR-06[리빙랩]헬스케어 제품화 실증 지원 기능<nl><ucel><ucel><fcel>LP-SFR-07[리빙랩]리빙랩 내 정보전달 기능<nl><ucel><ucel><fcel>LP-SFR-08[리빙랩]커뮤니티 기능<nl><ucel><ucel><fcel>LP-SFR-09[리빙랩]리빙랩 구축 요구 사항<nl><fcel>성능요구사항<fcel>PER-01<fcel>[공통]성능 요건 일반사항<nl><fcel>인터페이스요구사항<fcel>SIR-01<fcel>[공통]사용자 인터페이스 일반사항<nl><ucel><fcel>SIR-02<fcel>[공통]시스템 인터페이스 요구사항<nl><fcel>데이터품질요구사항<fcel>DAR-01<fcel>[공통]데이터 표준화<nl><ucel><fcel>DAR-02<fcel>[공통]데이터 표준 준수(공통 데이터 표준화)<nl><ucel><fcel>DAR-03<fcel>[공통]데이터 표준 관리<nl><ucel><fcel>DAR-04<fcel>[공통]데이터 구조 설계<nl><ucel><fcel>DAR-05<fcel>[공통]데이터 구조 검증<nl><ucel><fcel>DAR-06<fcel>[공통]데이터 구조 관리<nl><ucel><fcel>DAR-07<fcel>[공통]데이터 값 검증<nl><ucel><fcel>DAR-08<fcel>[공통]데이터 품질관리 체계<nl><ucel><fcel>DAR-09<fcel>[공통]데이터 개방 및 메타데이터 관리체계<nl><fcel>테스트요구사항<fcel>TER-01<fcel>[공통]단위테스트 요구사항<nl><ucel><fcel>TER-02<fcel>[공통]통합테스트 요구사항<nl><fcel>품질요구사항<fcel>QUR-01<fcel>[공통]시스템 품질 요구사항<nl><ucel><fcel>QUR-02<fcel>[공통]검수 요구사항<nl><fcel>프로젝트지원요구사항<fcel>PSR-01<fcel>[공통]교육 및 운영 지원 요구사항<nl><ucel><fcel>PSR-02<fcel>[공통]하자보수 요구사항<nl></otsl>"
stream = StringIO(otsl_text)
table_tag = DocTagsDocument.from_doctags_and_image_pairs(stream, images=None)
doc = DoclingDocument.load_from_doctags(table_tag)
for item in doc.tables:
    print(item.export_to_html(doc=doc))

import regex
from io import BytesIO
from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


class HTMLToDogTags:
    def __init__(
        self,
    ):
        self.backend_class = HTMLDocumentBackend
        self.format = InputFormat.HTML
        self.otsl_pattern = regex.compile(
            r"<otsl>.*?</otsl>",
            regex.DOTALL,
        )

    def extract_otsl(
        self,
        text: str,
    ) -> str:
        """Find the content <otsl>...</otsl>
        """
        if not isinstance(text, str):
            return None
        match = regex.search(
            self.otsl_pattern,
            text,
        )
        if match:
            return match.group(0).strip()
        else:
            return None

    def convert(
        self,
        html: str,
    ) -> str:
        html_bytes = html.encode("utf-8")
        bytes_io = BytesIO(html_bytes)
        in_doc = InputDocument(
            path_or_stream=bytes_io,
            format=self.format,
            backend=self.backend_class,
            filename="temp.html",
        )
        backend = self.backend_class(
            in_doc=in_doc,
            path_or_stream=bytes_io,
        )
        dl_document = backend.convert()
        doctags = dl_document.export_to_doctags()
        return self.extract_otsl(
            doctags,
        )
converter = HTMLToDogTags()
html = search_results.iloc[10].to_dict()["label"]
converter.convert(
    html,
)