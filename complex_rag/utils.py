from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from lightrag.operate import chunking_by_token_size
from lightrag.utils import logger


PAGE_MARKER = "<|complex_rag_page|>"


def normalize_caption_list(value: Any) -> list[str]:
    """将 caption 或 footnote 统一成非空字符串列表。 """

    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def get_table_body(item: dict[str, Any]) -> Any:
    """读取 MinerU 及常见别名中的表格正文。"""

    if item.get("table_body") not in (None, ""):
        return item["table_body"]
    if item.get("table_data") not in (None, ""):
        return item["table_data"]
    return item.get("text", "")


def format_table_body(table_body: Any) -> str:
    """将表格内容转换成适合 Prompt 和 chunk 的字符串。"""

    if isinstance(table_body, str):
        return table_body
    if isinstance(table_body, list):
        if not table_body:
            return ""
        if all(isinstance(row, (list, tuple)) for row in table_body):
            rows = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in table_body]
            column_count = max(len(row) for row in table_body)
            rows.insert(1, "| " + " | ".join(["---"] * column_count) + " |")
            return "\n".join(rows)
        return "\n".join(str(row) for row in table_body)
    return str(table_body)


def encode_image_to_base64(image_path: str | Path) -> str:
    """读取图片并编码为视觉模型可以接收的 base64 字符串。"""

    with Path(image_path).open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def separate_content(
    content_list: list[dict[str, Any]],
    source_label: str = "unknown_source",
) -> tuple[str, list[dict[str, Any]]]:
    """将 MinerU 内容列表拆成按页正文和多模态条目。"""

    text_by_page: dict[int, list[str]] = {}
    multimodal_items: list[dict[str, Any]] = []

    for index, item in enumerate(content_list):
        content_type = item.get("type", "text")

        if content_type == "text":
            text = item.get("text", "")
            if text.strip():
                page_idx = int(item.get("page_idx", 0))
                text_by_page.setdefault(page_idx, []).append(text.strip())
        else:
            multimodal_item = dict(item)
            multimodal_item.setdefault("_content_list_index", index)
            multimodal_items.append(multimodal_item)

    text_parts: list[str] = []
    for page_idx, page_parts in sorted(text_by_page.items()):
        marker_data = json.dumps(
            {"page_idx": page_idx, "source": source_label},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        text_parts.append(
            f"{PAGE_MARKER}{marker_data}\n" + "\n\n".join(page_parts)
        )
    text_content = "\n".join(text_parts)

    logger.info("Content separation complete:")
    logger.info("  - Text content length: %s characters", len(text_content))
    logger.info("  - Multimodal items count: %s", len(multimodal_items))

    return text_content, multimodal_items


def chunking_by_page(
    tokenizer: Any,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
) -> list[dict[str, Any]]:
    """按 MinerU 页边界分块，并为每个正文 chunk 保留页码。"""

    if PAGE_MARKER not in content:
        return chunking_by_token_size(
            tokenizer,
            content,
            split_by_character,
            split_by_character_only,
            chunk_overlap_token_size,
            chunk_token_size,
        )

    chunks: list[dict[str, Any]] = []
    for page_part in content.split(PAGE_MARKER):
        if not page_part.strip():
            continue

        marker_line, separator, page_text = page_part.partition("\n")
        if not separator:
            continue
        try:
            marker_data = json.loads(marker_line)
            page_idx = int(marker_data["page_idx"])
            source = str(marker_data.get("source") or "unknown_source")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning("Ignoring invalid page marker during text chunking")
            continue

        page_number = page_idx + 1
        header = f"[来源文档：{source}；页码：{page_number}]\n"
        header_tokens = len(tokenizer.encode(header))
        page_chunk_token_size = max(1, chunk_token_size - header_tokens)
        page_overlap = min(
            chunk_overlap_token_size,
            max(0, page_chunk_token_size - 1),
        )
        page_chunks = chunking_by_token_size(
            tokenizer,
            page_text,
            split_by_character,
            split_by_character_only,
            page_overlap,
            page_chunk_token_size,
        )
        for page_chunk in page_chunks:
            chunk_content = f"{header}{page_chunk['content']}"
            chunks.append(
                {
                    **page_chunk,
                    "content": chunk_content,
                    "tokens": len(tokenizer.encode(chunk_content)),
                    "chunk_order_index": len(chunks),
                    "page_idx": page_idx,
                }
            )

    return chunks


async def insert_text_content(
    lightrag: Any,
    input: str | list[str],
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
) -> None:
    """调用 LightRAG 的异步接口写入纯文本。"""

    logger.info("Starting text content insertion into LightRAG...")
    await lightrag.ainsert(
        input=input,
        file_paths=file_paths,
        split_by_character=split_by_character,
        split_by_character_only=split_by_character_only,
        ids=ids,
    )
    logger.info("Text content insertion complete")


def get_processor_for_type(
    modal_processors: dict[str, Any], content_type: str
) -> Any:
    """按内容类型选择图片或表格处理器。"""

    if content_type in {"image", "chart"}:
        return modal_processors.get("image")
    if content_type == "table":
        return modal_processors.get("table")
    return None
