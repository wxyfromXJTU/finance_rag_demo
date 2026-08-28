from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

# 在导入 LightRAG 前读取 .env，确保其内部配置可见。
load_dotenv(dotenv_path=".env", override=True)

from lightrag import LightRAG  
from lightrag.kg.shared_storage import initialize_pipeline_status  
from lightrag.utils import logger 

from complex_rag.config import RAGAnythingConfig  
from complex_rag.modalprocessors import (
    ContextConfig,
    ContextExtractor,
    ImageModalProcessor,
    TableModalProcessor,
)
from complex_rag.parser import MineruParser
from complex_rag.processor import ProcessorMixin
from complex_rag.query import FINAL_TOP_K, RETRIEVAL_CANDIDATE_K, QueryMixin
from complex_rag.utils import chunking_by_page


@dataclass
class RAGAnything(QueryMixin, ProcessorMixin):
    """管理配置、模型函数和 LightRAG storage 生命周期。"""

    lightrag: LightRAG | None = None
    llm_model_func: Callable[..., Any] | None = None
    vision_model_func: Callable[..., Any] | None = None
    embedding_func: Callable[..., Any] | None = None
    rerank_model_func: Callable[..., Any] | None = None
    config: RAGAnythingConfig | None = None
    lightrag_kwargs: dict[str, Any] = field(default_factory=dict)
    modal_processors: dict[str, Any] = field(default_factory=dict, init=False)
    context_extractor: ContextExtractor | None = field(default=None, init=False)
    _lightrag_rerank_func: Callable[..., Any] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:

        if self.config is None:
            self.config = RAGAnythingConfig()

        self.working_dir = self.config.working_dir
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)

        if self.config.parser != "mineru":
            raise ValueError(
                f"Unsupported parser: {self.config.parser}. demo only supports mineru."
            )
        self.doc_parser = MineruParser()

    def check_parser_installation(self) -> bool:
        """检查当前配置的 MinerU CLI 是否可用。"""

        return self.doc_parser.check_installation()

    async def _ensure_lightrag_initialized(self) -> dict[str, Any]:
        """确保 LightRAG 及共享 pipeline 状态已经初始化。"""

        try:
            if self.lightrag is not None:
                self._inherit_model_functions()

                storage_status = getattr(self.lightrag, "_storages_status", None)
                if getattr(storage_status, "name", None) != "INITIALIZED":
                    await self.lightrag.initialize_storages()
                    await initialize_pipeline_status()

                if not self.modal_processors:
                    self._initialize_processors()

                return {"success": True}

            if self.llm_model_func is None:
                return {
                    "success": False,
                    "error": "llm_model_func must be provided when LightRAG is not pre-initialized",
                }

            if self.embedding_func is None:
                return {
                    "success": False,
                    "error": "embedding_func must be provided when LightRAG is not pre-initialized",
                }

            params: dict[str, Any] = {
                "working_dir": self.working_dir,
                "llm_model_func": self.llm_model_func,
                "embedding_func": self.embedding_func,
                "chunking_func": chunking_by_page,
            }
            if self.rerank_model_func is not None:
                self._lightrag_rerank_func = self._build_rerank_model_func()
                params["rerank_model_func"] = self._lightrag_rerank_func
            params.update(self.lightrag_kwargs)

            self.lightrag = LightRAG(**params)
            await self.lightrag.initialize_storages()
            await initialize_pipeline_status()
            self._initialize_processors()
            logger.info("LightRAG storages initialized: %s", self.working_dir)
            return {"success": True}
        except Exception as exc:
            logger.error("Failed to initialize LightRAG: %s", exc, exc_info=True)
            return {"success": False, "error": str(exc)}

    def _inherit_model_functions(self) -> None:
        """从外部传入的 LightRAG 实例继承未显式提供的模型函数。"""

        if self.lightrag is None:
            return

        if self.llm_model_func is None:
            self.llm_model_func = getattr(self.lightrag, "llm_model_func", None)
        if self.embedding_func is None:
            self.embedding_func = getattr(self.lightrag, "embedding_func", None)
        if self.rerank_model_func is None:
            self.rerank_model_func = getattr(
                self.lightrag,
                "rerank_model_func",
                None,
            )
        if self.rerank_model_func is not None:
            self._lightrag_rerank_func = self._build_rerank_model_func()
            self.lightrag.rerank_model_func = self._lightrag_rerank_func

    def _build_rerank_model_func(self) -> Callable[..., Any]:
        """创建只处理前 30 条候选并返回 Top 5 的纯函数包装。"""

        if self.rerank_model_func is None:
            raise RuntimeError("rerank_model_func is not configured")
        provider = self.rerank_model_func

        async def rerank_documents(
            query: str,
            documents: list[str],
            top_n: int | None = None,
        ) -> list[dict[str, Any]]:
            candidates = documents[:RETRIEVAL_CANDIDATE_K]
            if not candidates:
                return []
            result = provider(
                query=query,
                documents=candidates,
                top_n=min(FINAL_TOP_K, len(candidates)),
            )
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, list) or not result:
                raise ValueError("Reranker returned no results")
            return result[:FINAL_TOP_K]

        return rerank_documents

    def _initialize_processors(self) -> None:
        """根据配置创建第一版图片和表格处理器。

        该方法必须在 LightRAG storage 初始化后执行，因为处理器会直接引用
        tokenizer、文本 chunk、向量库和知识图谱等 LightRAG 内部对象。
        图片优先使用 ``vision_model_func``，没有单独提供时回退到文本模型函数。
        """

        if self.lightrag is None:
            raise ValueError("LightRAG must be initialized before processors")

        self.context_extractor = ContextExtractor(
            config=ContextConfig(
                context_window=self.config.context_window,
                context_mode=self.config.context_mode,
                max_context_tokens=self.config.max_context_tokens,
                include_headers=self.config.include_headers,
                include_captions=self.config.include_captions,
            ),
            tokenizer=self.lightrag.tokenizer,
        )
        self.modal_processors = {}

        if self.config.enable_image_processing:
            self.modal_processors["image"] = ImageModalProcessor(
                lightrag=self.lightrag,
                modal_caption_func=self.vision_model_func or self.llm_model_func,
                context_extractor=self.context_extractor,
            )
        if self.config.enable_table_processing:
            self.modal_processors["table"] = TableModalProcessor(
                lightrag=self.lightrag,
                modal_caption_func=self.llm_model_func,
                context_extractor=self.context_extractor,
            )

    def set_content_source_for_context(
        self,
        content_source: list[dict[str, Any]],
    ) -> None:
        """向所有多模态处理器提供完整 MinerU 内容列表。"""

        for processor in self.modal_processors.values():
            processor.set_content_source(content_source, "minerU")

    async def finalize_storages(self) -> None:
        """将 LightRAG storage 刷新到磁盘并释放资源。"""

        if self.lightrag is None:
            return

        finalize = getattr(self.lightrag, "finalize_storages", None)
        if finalize is None:
            return

        await finalize()
        logger.info("LightRAG storages finalized")
