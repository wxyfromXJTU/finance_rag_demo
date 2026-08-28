from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from functools import partial
from pathlib import Path

import httpx
from dotenv import load_dotenv

# 直接执行脚本时，把项目根目录加入模块搜索路径。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from complex_rag import RAGAnything, RAGAnythingConfig


def _configure_logging(verbose: bool) -> None:
    """默认只显示警告和错误，排障时再开启过程日志。"""

    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    logging.getLogger().setLevel(level)
    for logger_name in (
        "complex_rag",
        "lightrag",
        "openai",
        "openai._base_client",
        "httpx",
        "httpcore",
    ):
        logging.getLogger(logger_name).setLevel(level)


def _required_env(name: str) -> str:
    """读取索引所需环境变量，缺失时给出明确错误。"""

    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


async def _call_rerank_api(
    query: str,
    documents: list[str],
    model: str,
    base_url: str,
    api_key: str,
    top_n: int | None,
) -> list[dict[str, int | float]]:
    """通过支持系统代理的 HTTP 客户端调用 AIHubMix 重排接口。"""

    # 与 LLM 请求一样读取 VPN 或系统提供的代理环境。
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(60.0, connect=20.0)
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        trust_env=True,
    ) as client:
        for attempt in range(3):
            try:
                response = await client.post(base_url, headers=headers, json=payload)
                response.raise_for_status()
                break
            except httpx.TransportError:
                if attempt == 2:
                    raise
                await asyncio.sleep(2**attempt)

    data = response.json()
    results = data.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Reranker response field 'results' must be a list")
    return [
        {
            "index": int(item["index"]),
            "relevance_score": float(item["relevance_score"]),
        }
        for item in results
    ]


def _build_rag() -> RAGAnything:
    """根据 .env 创建第一版 RAGAnything 及模型函数。"""

    api_key = _required_env("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or None
    llm_model = _required_env("LLM_MODEL")
    vision_model = _required_env("VISION_MODEL")
    embedding_model = _required_env("EMBEDDING_MODEL")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))
    rerank_model = os.getenv("RERANK_MODEL", "").strip()
    rerank_base_url = os.getenv("RERANK_BASE_URL", "").strip()
    if bool(rerank_model) != bool(rerank_base_url):
        raise ValueError(
            "RERANK_MODEL and RERANK_BASE_URL must be configured together"
        )
    rerank_api_key = os.getenv("RERANK_API_KEY", "").strip() or api_key

    def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        **kwargs,
    ):
        return openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    def vision_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        image_data: str | None = None,
        messages: list[dict] | None = None,
        **kwargs,
    ):
        if messages:
            return openai_complete_if_cache(
                vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        if not image_data:
            return llm_model_func(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        },
                    },
                ],
            }
        ]
        return openai_complete_if_cache(
            vision_model,
            "",
            system_prompt=system_prompt,
            history_messages=[],
            messages=messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=partial(
            openai_embed.func,
            model=embedding_model,
            api_key=api_key,
            base_url=base_url,
        ),
    )

    rerank_model_func = None
    if rerank_model and rerank_base_url:

        async def rerank_model_func(
            query: str,
            documents: list[str],
            top_n: int | None = None,
        ):
            return await _call_rerank_api(
                query=query,
                documents=documents,
                model=rerank_model,
                base_url=rerank_base_url,
                api_key=rerank_api_key,
                top_n=top_n,
            )

    return RAGAnything(
        config=RAGAnythingConfig(),
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        rerank_model_func=rerank_model_func,
    )


async def main() -> None:
    """执行一个 PDF 的完整索引并输出阶段统计。"""

    argument_parser = argparse.ArgumentParser(description="Index one PDF document")
    argument_parser.add_argument("pdf_path", help="待索引 PDF 的本地路径")
    argument_parser.add_argument("--lang", default="ch", help="MinerU 文档语言")
    argument_parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细过程日志",
    )
    args = argument_parser.parse_args()

    _configure_logging(args.verbose)
    rag = _build_rag()
    try:
        result = await rag.process_document_complete(
            args.pdf_path,
            lang=args.lang,
        )
        multimodal = result["multimodal"]
        print(
            f"Indexed {result['file_path']} as {result['doc_id']} "
            f"({result['content_blocks']} blocks)"
        )
        print(f"Text inserted: {result['text_inserted']}")
        print(
            "Multimodal: "
            f"processed={multimodal['processed']} "
            f"skipped={multimodal['skipped']} failed={multimodal['failed']}"
        )
        for failure in multimodal["failures"]:
            print(
                f"Multimodal failure #{failure['index']} "
                f"({failure['type']}): {failure['error']}",
                file=sys.stderr,
            )
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
