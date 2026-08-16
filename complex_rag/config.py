'''管理 complex_rag 自身配置'''
from dataclasses import dataclass, field
from lightrag.utils import get_env_value


@dataclass
class RAGAnythingConfig:

    # LightRAG 的工作目录，用于保存向量、图谱、KV 和文档状态等运行数据。
    working_dir: str = field(
        default=get_env_value("WORKING_DIR", "./rag_storage", str)
    )

    # Parser 的输出目录。目录用于保存 content_list、
    # Markdown、页面图片和从 PDF 中提取的图片等解析结果。
    parser_output_dir: str = field(
        default=get_env_value("OUTPUT_DIR", "./output", str)
    )

    # PDF 解析方式。
    parse_method: str = field(default=get_env_value("PARSE_METHOD", "auto", str))

    # 使用的文档解析器。
    parser: str = field(default=get_env_value("PARSER", "mineru", str))

    # 是否处理 MinerU 输出的 image 条目。
    enable_image_processing: bool = field(
        default=get_env_value("ENABLE_IMAGE_PROCESSING", True, bool)
    )

    # 是否处理 MinerU 输出的 table 条目。
    enable_table_processing: bool = field(
        default=get_env_value("ENABLE_TABLE_PROCESSING", True, bool)
    )

    # 提取多模态条目上下文时，向前和向后各包含多少页或内容块。
    context_window: int = field(default=get_env_value("CONTEXT_WINDOW", 1, int))

    # 上下文提取方式：page 按页提取，chunk 按相邻内容块提取。
    context_mode: str = field(default=get_env_value("CONTEXT_MODE", "page", str))

    # 单个多模态条目允许携带的最大上下文 token 数，超出部分会被截断。
    max_context_tokens: int = field(
        default=get_env_value("MAX_CONTEXT_TOKENS", 2000, int)
    )

    # 提取上下文时是否包含章节标题和页面标题。
    include_headers: bool = field(
        default=get_env_value("INCLUDE_HEADERS", True, bool)
    )

    # 提取上下文时是否包含图片或表格自身的 caption。
    include_captions: bool = field(
        default=get_env_value("INCLUDE_CAPTIONS", True, bool)
    )

    # 写入 LightRAG 的来源路径格式：False 只保存文件名，True 保存完整路径。
    use_full_path: bool = field(
        default=get_env_value("USE_FULL_PATH", False, bool)
    )
