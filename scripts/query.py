from __future__ import annotations

import argparse
import asyncio

from index import _build_rag


async def main() -> None:
    """查询已有 LightRAG storage，不重复索引文档。"""

    argument_parser = argparse.ArgumentParser(description="Query indexed documents")
    argument_parser.add_argument("question", help="要查询的问题")
    args = argument_parser.parse_args()

    rag = _build_rag()
    try:
        answer = await rag.aquery(args.question, mode="mix")
        print(answer)
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
