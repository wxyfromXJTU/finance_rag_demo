from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from lightrag.base import DocStatus
from lightrag.kg.shared_storage import get_namespace_data, get_pipeline_status_lock
from lightrag.operate import extract_entities, merge_nodes_and_edges
from lightrag.utils import compute_mdhash_id, logger

from complex_rag.prompt import IMAGE_CHUNK_TEMPLATE, TABLE_CHUNK_TEMPLATE
from complex_rag.utils import (
    format_table_body,
    get_processor_for_type,
    get_table_body,
    insert_text_content,
    normalize_caption_list,
    separate_content,
)


class ProcessorMixin:
    """串联 PDF 解析、文本入库和多模态批处理。"""

    def _get_file_reference(self, file_path: str | Path) -> str:
        """根据配置返回完整来源路径或文件名。"""

        if self.config.use_full_path:
            return str(file_path)
        return Path(file_path).name

    @staticmethod
    def _generate_content_based_doc_id(
        content_list: list[dict[str, Any]],
    ) -> str:
        """沿用参考项目规则，根据解析内容生成稳定文档 ID。"""

        content_hash_data: list[str] = []
        for item in content_list:
            content_type = item.get("type")
            if content_type == "text" and item.get("text"):
                content_hash_data.append(str(item["text"]).strip())
            elif content_type in {"image", "chart"} and item.get("img_path"):
                content_hash_data.append(f"image:{item['img_path']}")
            elif content_type == "table" and item.get("table_body"):
                content_hash_data.append(f"table:{item['table_body']}")
            else:
                content_hash_data.append(str(item))

        return compute_mdhash_id("\n".join(content_hash_data), prefix="doc-")

    async def parse_document(
        self,
        file_path: str | Path,
        output_dir: str | None = None,
        parse_method: str | None = None,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], str]:
        """使用 MinerU 解析单个 PDF，并返回内容列表和文档 ID。"""

        source_path = Path(file_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"PDF file does not exist: {source_path}")
        if source_path.suffix.lower() != ".pdf":
            raise ValueError("only supports PDF documents")

        parser_output_dir = output_dir or self.config.parser_output_dir
        method = parse_method or self.config.parse_method
        logger.info("[parse] Starting MinerU parsing: %s", source_path)
        content_list = await asyncio.to_thread(
            self.doc_parser.parse_pdf,
            pdf_path=source_path,
            output_dir=parser_output_dir,
            method=method,
            **kwargs,
        )
        if not content_list:
            raise ValueError("Parsing failed: no content was extracted")

        doc_id = self._generate_content_based_doc_id(content_list)
        logger.info("[parse] Parsed %s content blocks", len(content_list))
        return content_list, doc_id

    def _apply_chunk_template(
        self,
        content_type: str,
        original_item: dict[str, Any],
        description: str,
    ) -> str:
        """按类型把原始条目和模型描述整理成 LightRAG chunk。"""

        if content_type in {"image", "chart"}:
            captions = normalize_caption_list(
                original_item.get(
                    "image_caption",
                    original_item.get("img_caption"),
                )
            )
            footnotes = normalize_caption_list(
                original_item.get(
                    "image_footnote",
                    original_item.get("img_footnote"),
                )
            )
            return IMAGE_CHUNK_TEMPLATE.format(
                section_path=original_item.get("_section_path") or "None",
                neighbor_text=original_item.get("_neighbor_text") or "None",
                image_path=original_item.get("img_path") or "None",
                captions=", ".join(captions) if captions else "None",
                footnotes=", ".join(footnotes) if footnotes else "None",
                enhanced_caption=description,
            )

        if content_type == "table":
            table_caption = normalize_caption_list(
                original_item.get("table_caption")
            )
            table_footnote = normalize_caption_list(
                original_item.get("table_footnote")
            )
            return TABLE_CHUNK_TEMPLATE.format(
                table_img_path=original_item.get("img_path") or "None",
                table_caption=(
                    ", ".join(table_caption) if table_caption else "None"
                ),
                table_body=format_table_body(get_table_body(original_item)),
                table_footnote=(
                    ", ".join(table_footnote) if table_footnote else "None"
                ),
                enhanced_caption=description,
            )

        raise ValueError(f"Unsupported multimodal type: {content_type}")

    async def _generate_multimodal_descriptions(
        self,
        multimodal_items: list[dict[str, Any]],
        file_path: str,
        existing_chunks_count: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """并发生成图片和表格描述，单条失败不影响其他条目。"""

        semaphore = asyncio.Semaphore(
            max(1, int(getattr(self.lightrag, "max_parallel_insert", 2)))
        )

        async def process_item(item: dict[str, Any], index: int) -> dict[str, Any]:
            content_type = str(item.get("type", "unknown"))
            processor = get_processor_for_type(self.modal_processors, content_type)
            if processor is None:
                logger.debug(
                    "[multimodal] Skipping item %s with unsupported type: %s",
                    index,
                    content_type,
                )
                return {
                    "status": "skipped",
                    "index": index,
                    "content_type": content_type,
                }

            item_info = {
                "page_idx": item.get("page_idx", 0),
                "index": item.get("_content_list_index", index),
                "type": content_type,
            }
            try:
                async with semaphore:
                    description, entity_info = (
                        await processor.generate_description_only(
                            modal_content=item,
                            content_type=content_type,
                            item_info=item_info,
                            entity_name=None,
                        )
                    )
                return {
                    "status": "processed",
                    "index": index,
                    "content_type": content_type,
                    "description": description,
                    "entity_info": entity_info,
                    "original_item": item,
                    "item_info": item_info,
                    "chunk_order_index": existing_chunks_count + index,
                    "file_path": file_path,
                }
            except Exception as exc:
                logger.error(
                    "[multimodal] Failed item %s (%s): %s",
                    index,
                    content_type,
                    exc,
                )
                return {
                    "status": "failed",
                    "index": index,
                    "content_type": content_type,
                    "error": str(exc),
                }

        results = await asyncio.gather(
            *(process_item(item, index) for index, item in enumerate(multimodal_items))
        )
        processed_items = [
            result for result in results if result["status"] == "processed"
        ]
        summary = {
            "total": len(multimodal_items),
            "processed": len(processed_items),
            "skipped": sum(result["status"] == "skipped" for result in results),
            "failed": sum(result["status"] == "failed" for result in results),
            "failures": [
                {
                    "index": result["index"],
                    "type": result["content_type"],
                    "error": result["error"],
                }
                for result in results
                if result["status"] == "failed"
            ],
        }
        return processed_items, summary

    def _convert_to_lightrag_chunks(
        self,
        multimodal_data_list: list[dict[str, Any]],
        file_path: str,
        doc_id: str,
    ) -> dict[str, dict[str, Any]]:
        """批量构造符合 LightRAG 字段约定的多模态 chunks。"""

        file_reference = self._get_file_reference(file_path)
        chunks: dict[str, dict[str, Any]] = {}
        for data in multimodal_data_list:
            content = self._apply_chunk_template(
                data["content_type"],
                data["original_item"],
                data["description"],
            )
            chunk_id = compute_mdhash_id(content, prefix="chunk-")
            chunks[chunk_id] = {
                "content": content,
                "tokens": len(self.lightrag.tokenizer.encode(content)),
                "full_doc_id": doc_id,
                "chunk_order_index": data["chunk_order_index"],
                "file_path": file_reference,
                "llm_cache_list": [],
                "is_multimodal": True,
                "modal_entity_name": data["entity_info"]["entity_name"],
                "original_type": data["content_type"],
                "page_idx": data["item_info"].get("page_idx", 0),
            }
            data["chunk_id"] = chunk_id
        return chunks

    async def _store_multimodal_chunks(
        self,
        chunks: dict[str, dict[str, Any]],
    ) -> None:
        """批量写入文本 chunk 存储和 chunk 向量库。"""

        await self.lightrag.text_chunks.upsert(chunks)
        await self.lightrag.chunks_vdb.upsert(chunks)

    async def _store_multimodal_main_entities(
        self,
        multimodal_data_list: list[dict[str, Any]],
        file_path: str,
        doc_id: str,
    ) -> None:
        """批量写入图片和表格主实体，并更新文档实体清单。"""

        file_reference = self._get_file_reference(file_path)
        entities: dict[str, dict[str, Any]] = {}
        for data in multimodal_data_list:
            entity_info = data["entity_info"]
            entity_name = entity_info["entity_name"]
            entity_data = {
                "entity_name": entity_name,
                "entity_type": entity_info.get(
                    "entity_type", data["content_type"]
                ),
                "content": entity_info.get("summary", data["description"]),
                "source_id": data["chunk_id"],
                "file_path": file_reference,
            }
            entities[compute_mdhash_id(entity_name, prefix="ent-")] = entity_data
            await self.lightrag.chunk_entity_relation_graph.upsert_node(
                entity_name,
                {
                    "entity_id": entity_name,
                    "entity_type": entity_data["entity_type"],
                    "description": entity_data["content"],
                    "source_id": entity_data["source_id"],
                    "file_path": file_reference,
                    "created_at": int(time.time()),
                },
            )

        await self.lightrag.entities_vdb.upsert(entities)

        current = await self.lightrag.full_entities.get_by_id(doc_id) or {}
        entity_names = list(current.get("entity_names", []))
        known_names = set(entity_names)
        for entity_data in entities.values():
            if entity_data["entity_name"] not in known_names:
                entity_names.append(entity_data["entity_name"])
                known_names.add(entity_data["entity_name"])
        await self.lightrag.full_entities.upsert(
            {
                doc_id: {
                    **current,
                    "entity_names": entity_names,
                    "count": len(entity_names),
                    "update_time": int(time.time()),
                }
            }
        )

    async def _batch_extract_entities(
        self,
        chunks: dict[str, dict[str, Any]],
    ) -> list[Any]:
        """调用 LightRAG 原生接口批量抽取实体和关系。"""

        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()
        return await extract_entities(
            chunks=chunks,
            global_config=asdict(self.lightrag),
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=self.lightrag.llm_response_cache,
            text_chunks_storage=self.lightrag.text_chunks,
        )

    @staticmethod
    def _batch_add_belongs_to_relations(
        chunk_results: list[Any],
        multimodal_data_list: list[dict[str, Any]],
    ) -> list[Any]:
        """为每个抽取实体补充指向多模态主实体的归属关系。"""

        chunk_to_data = {
            data["chunk_id"]: data for data in multimodal_data_list
        }
        for maybe_nodes, maybe_edges in chunk_results:
            chunk_id = None
            for node_list in maybe_nodes.values():
                if node_list:
                    chunk_id = node_list[0].get("source_id")
                    break
            if chunk_id not in chunk_to_data:
                continue

            data = chunk_to_data[chunk_id]
            modal_entity_name = data["entity_info"]["entity_name"]
            for entity_name in maybe_nodes:
                if entity_name == modal_entity_name:
                    continue
                relation = {
                    "src_id": entity_name,
                    "tgt_id": modal_entity_name,
                    "description": (
                        f"Entity {entity_name} belongs to {modal_entity_name}"
                    ),
                    "keywords": "belongs_to,part_of,contained_in",
                    "source_id": chunk_id,
                    "weight": 10.0,
                    "file_path": data["file_path"],
                }
                maybe_edges.setdefault((entity_name, modal_entity_name), []).append(
                    relation
                )
        return chunk_results

    async def _batch_merge_entities_and_relations(
        self,
        chunk_results: list[Any],
        file_path: str,
        doc_id: str,
    ) -> None:
        """调用 LightRAG 原生接口批量去重合并节点和边。"""

        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()
        await merge_nodes_and_edges(
            chunk_results=chunk_results,
            knowledge_graph_inst=self.lightrag.chunk_entity_relation_graph,
            entity_vdb=self.lightrag.entities_vdb,
            relationships_vdb=self.lightrag.relationships_vdb,
            global_config=asdict(self.lightrag),
            full_entities_storage=self.lightrag.full_entities,
            full_relations_storage=self.lightrag.full_relations,
            doc_id=doc_id,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=self.lightrag.llm_response_cache,
            entity_chunks_storage=self.lightrag.entity_chunks,
            relation_chunks_storage=self.lightrag.relation_chunks,
            current_file_number=1,
            total_files=1,
            file_path=self._get_file_reference(file_path),
        )
        await self.lightrag._insert_done()

    async def _update_doc_status_with_chunks(
        self,
        doc_id: str,
        file_path: str,
        chunk_ids: list[str],
    ) -> None:
        """把多模态 chunk 合并进 LightRAG 的文档状态。"""

        current = await self.lightrag.doc_status.get_by_id(doc_id) or {}
        existing_chunk_ids = list(current.get("chunks_list", []))
        updated_chunk_ids = existing_chunk_ids + chunk_ids
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
        await self.lightrag.doc_status.upsert(
            {
                doc_id: {
                    **current,
                    "status": DocStatus.PROCESSED,
                    "content_summary": current.get("content_summary", ""),
                    "content_length": current.get("content_length", 0),
                    "chunks_list": updated_chunk_ids,
                    "chunks_count": len(updated_chunk_ids),
                    "created_at": current.get("created_at", timestamp),
                    "updated_at": timestamp,
                    "file_path": current.get(
                        "file_path", self._get_file_reference(file_path)
                    ),
                    "error_msg": current.get("error_msg", ""),
                }
            }
        )
        await self.lightrag.doc_status.index_done_callback()

    async def _process_multimodal_content(
        self,
        multimodal_items: list[dict[str, Any]],
        file_path: str,
        doc_id: str,
    ) -> dict[str, Any]:
        """按参考项目七阶段批处理图片、图表和表格。"""

        if not multimodal_items:
            return {
                "total": 0,
                "processed": 0,
                "skipped": 0,
                "failed": 0,
                "failures": [],
                "chunk_ids": [],
            }

        current_status = await self.lightrag.doc_status.get_by_id(doc_id)
        existing_chunks_count = (
            current_status.get("chunks_count", 0) if current_status else 0
        )
        data_list, summary = await self._generate_multimodal_descriptions(
            multimodal_items,
            file_path,
            existing_chunks_count,
        )
        if not data_list:
            summary["chunk_ids"] = []
            return summary

        chunks = self._convert_to_lightrag_chunks(data_list, file_path, doc_id)
        await self._store_multimodal_chunks(chunks)
        await self._store_multimodal_main_entities(data_list, file_path, doc_id)
        chunk_results = await self._batch_extract_entities(chunks)
        chunk_results = self._batch_add_belongs_to_relations(
            chunk_results,
            data_list,
        )
        await self._batch_merge_entities_and_relations(
            chunk_results,
            file_path,
            doc_id,
        )
        await self._update_doc_status_with_chunks(
            doc_id,
            file_path,
            list(chunks),
        )
        summary["chunk_ids"] = list(chunks)
        logger.info(
            "[multimodal] Batch complete: processed=%s skipped=%s failed=%s",
            summary["processed"],
            summary["skipped"],
            summary["failed"],
        )
        return summary

    async def process_document_complete(
        self,
        file_path: str | Path,
        output_dir: str | None = None,
        parse_method: str | None = None,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        doc_id: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """完成单个 PDF 的解析、文本入库和多模态批量入库。"""

        stage = "initialize"
        try:
            init_result = await self._ensure_lightrag_initialized()
            if not init_result.get("success"):
                raise RuntimeError(
                    f"LightRAG initialization failed: {init_result.get('error')}"
                )

            stage = "parse"
            if progress_callback is not None:
                progress_callback("开始解析")
            content_list, generated_doc_id = await self.parse_document(
                file_path,
                output_dir=output_dir,
                parse_method=parse_method,
                **kwargs,
            )
            actual_doc_id = doc_id or generated_doc_id
            file_reference = self._get_file_reference(file_path)
            text_content, multimodal_items = separate_content(
                content_list,
                source_label=file_reference,
            )
            if multimodal_items:
                self.set_content_source_for_context(content_list)

            stage = "text_insert"
            if progress_callback is not None:
                progress_callback("正文入库")
            text_inserted = bool(text_content.strip())
            if text_inserted:
                await insert_text_content(
                    self.lightrag,
                    input=text_content,
                    file_paths=file_reference,
                    split_by_character=split_by_character,
                    split_by_character_only=split_by_character_only,
                    ids=actual_doc_id,
                )

            stage = "multimodal"
            if progress_callback is not None:
                progress_callback("多模态处理")
            multimodal_summary = await self._process_multimodal_content(
                multimodal_items,
                str(file_path),
                actual_doc_id,
            )
            result = {
                "doc_id": actual_doc_id,
                "file_path": file_reference,
                "content_blocks": len(content_list),
                "text_inserted": text_inserted,
                "multimodal": multimodal_summary,
            }
            logger.info("Document processing complete: %s", result)
            return result
        except Exception as exc:
            logger.error(
                "Document processing failed at %s stage: %s",
                stage,
                exc,
            )
            raise
