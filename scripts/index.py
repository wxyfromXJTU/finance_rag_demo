from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from functools import partial
from pathlib import Path

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


def _build_rag() -> RAGAnything:
    """根据 .env 创建第一版 RAGAnything 及模型函数。"""

    api_key = _required_env("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or None
    llm_model = _required_env("LLM_MODEL")
    vision_model = _required_env("VISION_MODEL")
    embedding_model = _required_env("EMBEDDING_MODEL")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))

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
    return RAGAnything(
        config=RAGAnythingConfig(),
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
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
