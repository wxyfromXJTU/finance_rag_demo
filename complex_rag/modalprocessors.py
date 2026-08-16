from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from lightrag.kg.shared_storage import get_namespace_data, get_pipeline_status_lock
from lightrag.operate import extract_entities, merge_nodes_and_edges
from lightrag.utils import compute_mdhash_id

from complex_rag.prompt import (
    IMAGE_ANALYSIS_PROMPT,
    IMAGE_ANALYSIS_SYSTEM,
    IMAGE_CHUNK_TEMPLATE,
    TABLE_ANALYSIS_PROMPT,
    TABLE_ANALYSIS_SYSTEM,
    TABLE_CHUNK_TEMPLATE,
)
from complex_rag.utils import (
    encode_image_to_base64,
    format_table_body,
    get_table_body,
    normalize_caption_list,
)


@dataclass
class ContextConfig:
    """多模态条目附近正文的提取配置。"""

    context_window: int = 1
    context_mode: str = "page"
    max_context_tokens: int = 2000
    include_headers: bool = True
    include_captions: bool = True
    filter_content_types: list[str] = field(default_factory=lambda: ["text"])


class ContextExtractor:
    """从 MinerU content_list 中提取页级或相邻块级正文。"""

    def __init__(self, config: ContextConfig | None = None, tokenizer: Any = None):
        self.config = config or ContextConfig()
        self.tokenizer = tokenizer

    def extract_context(
        self,
        content_source: list[dict[str, Any]],
        current_item_info: dict[str, Any],
        content_format: str = "auto",
    ) -> str:
        """提取当前图片或表格周围的正文。"""

        if not content_source:
            return ""
        if self.config.context_mode == "chunk":
            return self._extract_chunk_context(content_source, current_item_info)
        return self._extract_page_context(content_source, current_item_info)

    def _extract_page_context(
        self,
        content_list: list[dict[str, Any]],
        current_item_info: dict[str, Any],
    ) -> str:
        """按页码窗口收集正文，并为非当前页文本添加页码标记。"""

        current_page = current_item_info.get("page_idx", 0)
        start_page = max(0, current_page - self.config.context_window)
        end_page = current_page + self.config.context_window
        parts: list[str] = []

        for item in content_list:
            item_page = item.get("page_idx", 0)
            if not start_page <= item_page <= end_page:
                continue
            text = self._extract_text_from_item(item)
            if text:
                if item_page != current_page:
                    text = f"[Page {item_page}] {text}"
                parts.append(text)

        return self._truncate_context("\n".join(parts))

    def _extract_chunk_context(
        self,
        content_list: list[dict[str, Any]],
        current_item_info: dict[str, Any],
    ) -> str:
        """按内容块索引收集前后正文，不包含当前多模态条目本身。"""

        current_index = current_item_info.get("index", 0)
        start_index = max(0, current_index - self.config.context_window)
        end_index = min(len(content_list), current_index + self.config.context_window + 1)
        parts = [
            self._extract_text_from_item(content_list[index])
            for index in range(start_index, end_index)
            if index != current_index
        ]
        return self._truncate_context("\n".join(text for text in parts if text))

    def _extract_text_from_item(self, item: dict[str, Any]) -> str:
        """从单个内容块提取可用于上下文的文本。"""

        item_type = item.get("type", "")
        if item_type not in self.config.filter_content_types:
            return ""
        if item_type == "text":
            text = str(item.get("text", "") or "").strip()
            level = int(item.get("text_level", 0) or 0)
            if text and self.config.include_headers and level > 0:
                return f"{'#' * level} {text}"
            return text
        if not self.config.include_captions:
            return ""
        if item_type in {"image", "chart"}:
            captions = normalize_caption_list(
                item.get("image_caption", item.get("img_caption"))
            )
            return f"[Image: {', '.join(captions)}]" if captions else ""
        if item_type == "table":
            captions = normalize_caption_list(item.get("table_caption"))
            return f"[Table: {', '.join(captions)}]" if captions else ""
        return ""

    def _truncate_context(self, context: str) -> str:
        """把上下文限制在 ``max_context_tokens`` 内。"""

        if not context:
            return ""
        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(context)
            if len(tokens) <= self.config.max_context_tokens:
                return context
            return self.tokenizer.decode(tokens[: self.config.max_context_tokens]) + "..."
        if len(context) <= self.config.max_context_tokens:
            return context
        return context[: self.config.max_context_tokens] + "..."


class BaseModalProcessor:
    """图片和表格处理器共用的模型响应解析与 LightRAG 写入逻辑。"""

    def __init__(
        self,
        lightrag: Any,
        modal_caption_func: Callable[..., Any],
        context_extractor: ContextExtractor | None = None,
    ) -> None:
        """初始化多模态处理器的公共依赖。"""

        if modal_caption_func is None:
            raise ValueError("modal_caption_func must be provided")
        self.lightrag = lightrag
        self.modal_caption_func = modal_caption_func
        self.context_extractor = context_extractor or ContextExtractor(
            tokenizer=lightrag.tokenizer
        )
        self.content_source: list[dict[str, Any]] | None = None
        self.content_format = "auto"

    def set_content_source(
        self, content_source: list[dict[str, Any]], content_format: str = "auto"
    ) -> None:
        """保存完整内容列表，供处理单个多模态条目时查找附近正文。"""

        self.content_source = content_source
        self.content_format = content_format

    def _get_context_for_item(self, item_info: dict[str, Any] | None) -> str:
        """根据当前条目的页码和索引取得上下文，没有来源时返回空文本。"""

        if not self.content_source or not item_info:
            return ""
        return self.context_extractor.extract_context(
            self.content_source,
            item_info,
            self.content_format,
        )

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """删除推理模型可能返回的 ``think`` 标签内容，避免污染入库文本。"""

        cleaned = re.sub(
            r"<think(?:ing)?>.*?</think(?:ing)?>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return cleaned.strip()

    def _parse_response(
        self,
        response: str,
        content_type: str,
        entity_name: str | None,
    ) -> tuple[str, dict[str, Any]]:
        """解析模型返回的描述和实体信息。"""

        cleaned = self._strip_thinking_tags(response)
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned).strip()

        try:
            response_data = json.loads(cleaned)
            description = response_data["detailed_description"]
            entity_info = response_data["entity_info"]
            if not all(
                entity_info.get(key)
                for key in ("entity_name", "entity_type", "summary")
            ):
                raise ValueError("entity_info is incomplete")
            if entity_name:
                entity_info["entity_name"] = entity_name
            else:
                entity_info["entity_name"] = (
                    f"{entity_info['entity_name']} ({entity_info['entity_type']})"
                )
            return str(description), entity_info
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            fallback_name = entity_name or f"{content_type}_{compute_mdhash_id(cleaned)}"
            summary = cleaned[:100] + ("..." if len(cleaned) > 100 else "")
            return cleaned, {
                "entity_name": fallback_name,
                "entity_type": content_type,
                "summary": summary,
            }

    async def _create_entity_and_chunk(
        self,
        modal_chunk: str,
        entity_info: dict[str, Any],
        file_path: str,
        batch_mode: bool = False,
        doc_id: str | None = None,
        chunk_order_index: int = 0,
    ) -> tuple[str, dict[str, Any], list[Any]]:
        """写入多模态主实体和 chunk，然后抽取并合并实体关系。"""

        chunk_id = compute_mdhash_id(modal_chunk, prefix="chunk-")
        tokens = len(self.lightrag.tokenizer.encode(modal_chunk))
        actual_doc_id = doc_id or chunk_id
        chunk_data = {
            "tokens": tokens,
            "content": modal_chunk,
            "chunk_order_index": chunk_order_index,
            "full_doc_id": actual_doc_id,
            "file_path": file_path,
        }

        # 保存完整 chunk，并写入 chunk 向量库以支持相似度召回。
        await self.lightrag.text_chunks.upsert({chunk_id: chunk_data})
        await self.lightrag.chunks_vdb.upsert({chunk_id: chunk_data})

        # 在知识图谱中创建由模型提炼出的图片或表格主实体。
        node_data = {
            "entity_id": entity_info["entity_name"],
            "entity_type": entity_info["entity_type"],
            "description": entity_info["summary"],
            "source_id": chunk_id,
            "file_path": file_path,
            "created_at": int(time.time()),
        }
        await self.lightrag.chunk_entity_relation_graph.upsert_node(
            entity_info["entity_name"], node_data
        )

        # 同步写入实体向量库，使实体名称和摘要可以参与检索。
        entity_id = compute_mdhash_id(entity_info["entity_name"], prefix="ent-")
        await self.lightrag.entities_vdb.upsert(
            {
                entity_id: {
                    "entity_name": entity_info["entity_name"],
                    "entity_type": entity_info["entity_type"],
                    "content": f"{entity_info['entity_name']}\n{entity_info['summary']}",
                    "source_id": chunk_id,
                    "file_path": file_path,
                }
            }
        )

        # 复用 LightRAG 的原生实体关系抽取与去重合并流程。
        chunk_results = await self._process_chunk_for_extraction(
            chunk_id,
            entity_info["entity_name"],
            batch_mode,
        )

        return entity_info["summary"], {
            "entity_name": entity_info["entity_name"],
            "entity_type": entity_info["entity_type"],
            "description": entity_info["summary"],
            "chunk_id": chunk_id,
        }, chunk_results

    async def _process_chunk_for_extraction(
        self,
        chunk_id: str,
        modal_entity_name: str,
        batch_mode: bool = False,
    ) -> list[Any]:
        """从多模态 chunk 抽取实体和关系，并交给 LightRAG 去重合并。"""

        chunk_data = await self.lightrag.text_chunks.get_by_id(chunk_id)
        if not chunk_data:
            raise ValueError(f"Chunk not found after insertion: {chunk_id}")

        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        # extract_entities 使用与普通文本入库相同的模型提示词和解析逻辑。
        chunk_results = await extract_entities(
            chunks={chunk_id: chunk_data},
            global_config=asdict(self.lightrag),
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=self.lightrag.llm_response_cache,
            text_chunks_storage=self.lightrag.text_chunks,
        )

        # 把 chunk 中抽取出的实体连接到图片或表格主实体。
        for maybe_nodes, maybe_edges in chunk_results:
            for entity_name in maybe_nodes:
                if entity_name == modal_entity_name:
                    continue
                relation_data = {
                    "src_id": entity_name,
                    "tgt_id": modal_entity_name,
                    "description": (
                        f"Entity {entity_name} belongs to {modal_entity_name}"
                    ),
                    "keywords": "belongs_to,part_of,contained_in",
                    "source_id": chunk_id,
                    "weight": 10.0,
                    "file_path": chunk_data.get("file_path", "manual_creation"),
                }
                maybe_edges.setdefault((entity_name, modal_entity_name), []).append(
                    relation_data
                )

        if not batch_mode:
            # merge_nodes_and_edges 负责合并同名实体、同端点关系及其来源和描述。
            await merge_nodes_and_edges(
                chunk_results=chunk_results,
                knowledge_graph_inst=self.lightrag.chunk_entity_relation_graph,
                entity_vdb=self.lightrag.entities_vdb,
                relationships_vdb=self.lightrag.relationships_vdb,
                global_config=asdict(self.lightrag),
                full_entities_storage=self.lightrag.full_entities,
                full_relations_storage=self.lightrag.full_relations,
                doc_id=chunk_data.get("full_doc_id"),
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.lightrag.llm_response_cache,
                entity_chunks_storage=self.lightrag.entity_chunks,
                relation_chunks_storage=self.lightrag.relation_chunks,
                current_file_number=1,
                total_files=1,
                file_path=chunk_data.get("file_path", "manual_creation"),
            )
            await self.lightrag._insert_done()

        return chunk_results


class ImageModalProcessor(BaseModalProcessor):
    """使用视觉模型分析 MinerU 图片或图表条目。"""

    async def generate_description_only(
        self,
        modal_content: dict[str, Any],
        content_type: str,
        item_info: dict[str, Any] | None = None,
        entity_name: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """生成图片描述和主实体，供批处理阶段统一入库。"""

        image_path = str(modal_content.get("img_path", ""))
        if not image_path or not Path(image_path).is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        captions = normalize_caption_list(
            modal_content.get("image_caption", modal_content.get("img_caption"))
        )
        footnotes = normalize_caption_list(
            modal_content.get("image_footnote", modal_content.get("img_footnote"))
        )
        context = self._get_context_for_item(item_info)
        prompt = IMAGE_ANALYSIS_PROMPT.format(
            context=context or "无",
            section_path=modal_content.get("_section_path") or "None",
            entity_name=entity_name or "为这张图片生成语义化名称",
            image_path=image_path,
            captions=captions or "None",
            footnotes=footnotes or "None",
        )
        response = await self.modal_caption_func(
            prompt,
            image_data=encode_image_to_base64(image_path),
            system_prompt=IMAGE_ANALYSIS_SYSTEM,
        )
        return self._parse_response(response, content_type, entity_name)

    async def process_multimodal_content(
        self,
        modal_content: dict[str, Any],
        content_type: str,
        file_path: str = "manual_creation",
        entity_name: str | None = None,
        item_info: dict[str, Any] | None = None,
        batch_mode: bool = False,
        doc_id: str | None = None,
        chunk_order_index: int = 0,
    ) -> tuple[str, dict[str, Any], list[Any]]:
        """分析图片或图表并写入 LightRAG。"""

        description, entity_info = await self.generate_description_only(
            modal_content,
            content_type,
            item_info,
            entity_name,
        )
        image_path = str(modal_content.get("img_path", ""))
        captions = normalize_caption_list(
            modal_content.get("image_caption", modal_content.get("img_caption"))
        )
        footnotes = normalize_caption_list(
            modal_content.get("image_footnote", modal_content.get("img_footnote"))
        )
        # 把原始元数据与视觉描述拼成最终可检索文本块。
        modal_chunk = IMAGE_CHUNK_TEMPLATE.format(
            section_path=modal_content.get("_section_path") or "None",
            neighbor_text=modal_content.get("_neighbor_text") or "None",
            image_path=image_path,
            captions=", ".join(captions) if captions else "None",
            footnotes=", ".join(footnotes) if footnotes else "None",
            enhanced_caption=description,
        )
        return await self._create_entity_and_chunk(
            modal_chunk,
            entity_info,
            file_path,
            batch_mode,
            doc_id,
            chunk_order_index,
        )


class TableModalProcessor(BaseModalProcessor):
    """使用文本模型分析 MinerU 表格条目。"""

    async def generate_description_only(
        self,
        modal_content: dict[str, Any],
        content_type: str,
        item_info: dict[str, Any] | None = None,
        entity_name: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """生成表格描述和主实体，供批处理阶段统一入库。"""

        table_caption = normalize_caption_list(modal_content.get("table_caption"))
        table_footnote = normalize_caption_list(modal_content.get("table_footnote"))
        table_body = format_table_body(get_table_body(modal_content))
        context = self._get_context_for_item(item_info)
        prompt = TABLE_ANALYSIS_PROMPT.format(
            context=context or "无",
            entity_name=entity_name or "为这张表格生成语义化名称",
            table_img_path=modal_content.get("img_path") or "None",
            table_caption=table_caption or "None",
            table_body=table_body,
            table_footnote=table_footnote or "None",
        )
        response = await self.modal_caption_func(
            prompt,
            system_prompt=TABLE_ANALYSIS_SYSTEM,
        )
        return self._parse_response(response, content_type, entity_name)

    async def process_multimodal_content(
        self,
        modal_content: dict[str, Any],
        content_type: str,
        file_path: str = "manual_creation",
        entity_name: str | None = None,
        item_info: dict[str, Any] | None = None,
        batch_mode: bool = False,
        doc_id: str | None = None,
        chunk_order_index: int = 0,
    ) -> tuple[str, dict[str, Any], list[Any]]:
        """分析结构化表格并写入 LightRAG。 """

        description, entity_info = await self.generate_description_only(
            modal_content,
            content_type,
            item_info,
            entity_name,
        )
        table_caption = normalize_caption_list(modal_content.get("table_caption"))
        table_footnote = normalize_caption_list(modal_content.get("table_footnote"))
        table_body = format_table_body(get_table_body(modal_content))
        # 同时保留原始表格结构和模型描述，方便精确数值与语义检索。
        modal_chunk = TABLE_CHUNK_TEMPLATE.format(
            table_img_path=modal_content.get("img_path") or "None",
            table_caption=", ".join(table_caption) if table_caption else "None",
            table_body=table_body,
            table_footnote=", ".join(table_footnote) if table_footnote else "None",
            enhanced_caption=description,
        )
        return await self._create_entity_and_chunk(
            modal_chunk,
            entity_info,
            file_path,
            batch_mode,
            doc_id,
            chunk_order_index,
        )
