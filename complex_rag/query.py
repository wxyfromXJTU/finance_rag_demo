from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lightrag import QueryParam
from lightrag.operate import kg_query
from lightrag.utils import logger

from complex_rag.prompt import VLM_QUERY_SYSTEM
from complex_rag.utils import encode_image_to_base64


class QueryMixin:
    """提供 LightRAG mix 查询和参考项目的 VLM 增强查询。"""

    async def aquery(
        self,
        query: str,
        mode: str = "mix",
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """执行查询；有视觉模型时自动尝试 VLM 增强。"""

        if self.vision_model_func is not None:
            return await self.aquery_vlm_enhanced(
                query,
                mode=mode,
                system_prompt=system_prompt,
                **kwargs,
            )

        init_result = await self._ensure_lightrag_initialized()
        if not init_result.get("success"):
            raise RuntimeError(
                f"LightRAG initialization failed: {init_result.get('error')}"
            )

        logger.info("Executing LightRAG query in %s mode", mode)
        return await self.lightrag.aquery(
            query,
            param=QueryParam(mode=mode, **kwargs),
            system_prompt=system_prompt,
        )

    async def aquery_vlm_enhanced(
        self,
        query: str,
        mode: str = "mix",
        system_prompt: str | None = None,
        extra_safe_dirs: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """先检索 LightRAG Prompt；召回有效图片时再调用 VLM 回答。"""

        if self.vision_model_func is None:
            raise ValueError("VLM enhanced query requires vision_model_func")

        init_result = await self._ensure_lightrag_initialized()
        if not init_result.get("success"):
            raise RuntimeError(
                f"LightRAG initialization failed: {init_result.get('error')}"
            )

        # 只取得 LightRAG 已组装的检索 Prompt，不先生成普通文本答案。
        retrieval_param = QueryParam(mode=mode, only_need_prompt=True, **kwargs)
        raw_prompt = await self.lightrag.aquery(query, param=retrieval_param)
        enhanced_prompt, images_found = self._process_image_paths_for_vlm(
            str(raw_prompt),
            extra_safe_dirs=extra_safe_dirs,
        )

        if images_found == 0:
            logger.info(
                "No valid images found; answering with the retrieved prompt"
            )
            return await self.llm_model_func(
                str(raw_prompt),
                system_prompt=system_prompt,
            )

        messages = self._build_vlm_messages_with_images(
            enhanced_prompt,
            query,
            system_prompt,
        )
        return await self._call_vlm_with_multimodal_content(messages)

    async def aquery_with_trace(
        self,
        query: str,
        mode: str = "mix",
        system_prompt: str | None = None,
        extra_safe_dirs: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """单次检索后返回最终答案和带页码的实际证据。"""

        init_result = await self._ensure_lightrag_initialized()
        if not init_result.get("success"):
            raise RuntimeError(
                f"LightRAG initialization failed: {init_result.get('error')}"
            )
        if mode != "mix":
            raise ValueError("Traced evaluation queries only support mix mode")

        query_param = QueryParam(
            mode=mode,
            only_need_prompt=True,
            stream=False,
            **kwargs,
        )
        try:
            query_result = await kg_query(
                query.strip(),
                self.lightrag.chunk_entity_relation_graph,
                self.lightrag.entities_vdb,
                self.lightrag.relationships_vdb,
                self.lightrag.text_chunks,
                query_param,
                asdict(self.lightrag),
                hashing_kv=self.lightrag.llm_response_cache,
                system_prompt=None,
                chunks_vdb=self.lightrag.chunks_vdb,
            )
        finally:
            await self.lightrag._query_done()

        if query_result is None:
            return {"answer": "未检索到相关证据。", "evidence": []}

        raw_prompt = str(query_result.content)
        raw_data = query_result.raw_data or {}
        evidence = await self._enrich_evidence(raw_data)
        enhanced_prompt, images_found = self._process_image_paths_for_vlm(
            raw_prompt,
            extra_safe_dirs=extra_safe_dirs,
        )
        if images_found and self.vision_model_func is not None:
            messages = self._build_vlm_messages_with_images(
                enhanced_prompt,
                query,
                system_prompt,
            )
            answer = await self._call_vlm_with_multimodal_content(messages)
        else:
            answer = await self.llm_model_func(
                raw_prompt,
                system_prompt=system_prompt,
            )

        return {"answer": str(answer), "evidence": evidence}

    async def _enrich_evidence(
        self,
        raw_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """用 chunk KV 中的原始字段补全检索结果。"""

        chunks = raw_data.get("data", {}).get("chunks", [])
        chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in chunks]
        stored_chunks = await self.lightrag.text_chunks.get_by_ids(chunk_ids)
        evidence: list[dict[str, Any]] = []
        for rank, (chunk, stored) in enumerate(
            zip(chunks, stored_chunks, strict=True),
            start=1,
        ):
            stored = stored or {}
            page_idx = stored.get("page_idx")
            evidence.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.get("chunk_id", ""),
                    "file_path": chunk.get("file_path", "unknown_source"),
                    "page_idx": page_idx,
                    "page_number": (
                        int(page_idx) + 1 if page_idx is not None else None
                    ),
                    "content": chunk.get("content", ""),
                    "is_multimodal": bool(stored.get("is_multimodal", False)),
                    "original_type": stored.get("original_type", "text"),
                }
            )
        return evidence

    def _process_image_paths_for_vlm(
        self,
        prompt: str,
        extra_safe_dirs: list[str] | None = None,
    ) -> tuple[str, int]:
        """校验检索 Prompt 中的图片路径，并按出现位置加入 VLM 标记。"""

        self._current_images_base64: list[str] = []
        images_processed = 0
        image_path_pattern = re.compile(
            r"(?:Image\s+Path|图片路径)\s*[:：]\s*"
            r"([^\r\n]*?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif))",
            flags=re.IGNORECASE,
        )

        safe_dirs = [Path.cwd().resolve()]
        if self.config is not None:
            safe_dirs.extend(
                [
                    Path(self.config.working_dir).resolve(),
                    Path(self.config.parser_output_dir).resolve(),
                ]
            )
        safe_dirs.extend(Path(path).resolve() for path in extra_safe_dirs or [])

        def replace_image_path(match: re.Match[str]) -> str:
            nonlocal images_processed

            image_path = Path(match.group(1).strip())
            if image_path.is_symlink():
                logger.warning("Blocked symlink image path: %s", image_path)
                return match.group(0)
            try:
                resolved_path = image_path.resolve(strict=True)
            except (OSError, RuntimeError):
                return match.group(0)

            is_safe = any(
                resolved_path == safe_dir or resolved_path.is_relative_to(safe_dir)
                for safe_dir in safe_dirs
            )
            valid_extensions = {
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".bmp",
                ".webp",
                ".tiff",
                ".tif",
            }
            if (
                not is_safe
                or not resolved_path.is_file()
                or resolved_path.suffix.lower() not in valid_extensions
                or resolved_path.stat().st_size > 50 * 1024 * 1024
            ):
                logger.warning("Blocked unsafe image path: %s", image_path)
                return match.group(0)

            image_base64 = encode_image_to_base64(resolved_path)
            self._current_images_base64.append(image_base64)
            images_processed += 1
            return f"{match.group(0)}\n[VLM_IMAGE_{images_processed}]"

        enhanced_prompt = image_path_pattern.sub(replace_image_path, prompt)
        return enhanced_prompt, images_processed

    def _build_vlm_messages_with_images(
        self,
        enhanced_prompt: str,
        user_query: str,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """根据 Prompt 中的标记，把文本和 base64 图片按原位置组成消息。"""

        images_base64 = getattr(self, "_current_images_base64", [])
        content_parts: list[dict[str, Any]] = []
        text_parts = enhanced_prompt.split("[VLM_IMAGE_")

        if text_parts[0].strip():
            content_parts.append({"type": "text", "text": text_parts[0]})
        for text_part in text_parts[1:]:
            marker_match = re.match(r"(\d+)\](.*)", text_part, re.DOTALL)
            if marker_match is None:
                continue
            image_index = int(marker_match.group(1)) - 1
            if 0 <= image_index < len(images_base64):
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/jpeg;base64,"
                                f"{images_base64[image_index]}"
                            )
                        },
                    }
                )
            remaining_text = marker_match.group(2)
            if remaining_text.strip():
                content_parts.append({"type": "text", "text": remaining_text})

        content_parts.append(
            {
                "type": "text",
                "text": f"\n用户问题：{user_query}\n请根据以上上下文和图片回答。",
            }
        )
        full_system_prompt = VLM_QUERY_SYSTEM
        if system_prompt:
            full_system_prompt = f"{full_system_prompt}\n{system_prompt}"
        return [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": content_parts},
        ]

    async def _call_vlm_with_multimodal_content(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        """把完整多模态消息交给外部注入的视觉模型函数。"""

        return await self.vision_model_func("", messages=messages)
