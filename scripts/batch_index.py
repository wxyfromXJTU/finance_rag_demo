from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from index import _build_rag, _configure_logging


def _completed_files(status_file: Path) -> set[str]:
    """读取已成功或部分完成的文件，避免断点续跑时重复写入。"""

    if not status_file.is_file():
        return set()

    completed: set[str] = set()
    for line in status_file.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") in {"success", "partial"}:
            completed.add(str(record.get("pdf_path", "")))
    return completed


def _append_status(status_file: Path, record: dict) -> None:
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with status_file.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


async def main() -> None:
    """按文件名顺序批量索引 PDF，并逐条保存运行状态。"""

    parser = argparse.ArgumentParser(description="Index all PDFs in one directory")
    parser.add_argument(
        "pdf_dir",
        nargs="?",
        default="data/pdf_ch",
        help="PDF 目录，默认 data/pdf_ch",
    )
    parser.add_argument("--lang", default="ch", help="MinerU 文档语言")
    parser.add_argument(
        "--status-file",
        default="experiment_results/index_status.jsonl",
        help="逐文档状态文件",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过状态文件中已完成（含多模态部分失败）的 PDF",
    )
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    args = parser.parse_args()

    _configure_logging(args.verbose)
    pdf_dir = Path(args.pdf_dir).resolve()
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"), key=lambda path: path.name)
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {pdf_dir}")

    status_file = Path(args.status_file).resolve()
    if status_file.exists() and not args.resume:
        raise FileExistsError(
            f"Status file already exists: {status_file}; "
            "use --resume or another --status-file"
        )
    working_dir = Path(os.getenv("WORKING_DIR", "./rag_storage")).resolve()
    existing_entries = (
        [entry for entry in working_dir.iterdir() if entry.name != ".gitkeep"]
        if working_dir.is_dir()
        else []
    )
    if existing_entries and not args.resume:
        raise FileExistsError(
            f"WORKING_DIR is not empty: {working_dir}; "
            "use a new empty WORKING_DIR for the formal experiment"
        )
    completed = _completed_files(status_file) if args.resume else set()
    rag = _build_rag()
    succeeded = partial = failed = skipped = 0
    try:
        for index, pdf_path in enumerate(pdf_files, start=1):
            resolved_path = str(pdf_path.resolve())
            if resolved_path in completed:
                skipped += 1
                print(f"[{index}/{len(pdf_files)}] skipped {pdf_path.name}")
                continue

            started_at = time.perf_counter()

            def show_progress(stage: str) -> None:
                print(
                    f"[{index}/{len(pdf_files)}] {pdf_path.name} | {stage}",
                    flush=True,
                )

            try:
                result = await rag.process_document_complete(
                    pdf_path,
                    lang=args.lang,
                    progress_callback=show_progress,
                )
                multimodal = result["multimodal"]
                status = "partial" if multimodal["failed"] else "success"
                if status == "partial":
                    partial += 1
                else:
                    succeeded += 1
                record = {
                    "pdf_path": resolved_path,
                    "status": status,
                    "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                    "result": result,
                }
            except Exception as exc:
                failed += 1
                status = "failed"
                record = {
                    "pdf_path": resolved_path,
                    "status": status,
                    "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                    "error": str(exc),
                }

            _append_status(status_file, record)
            elapsed = record["elapsed_seconds"]
            if status == "failed":
                print(
                    f"[{index}/{len(pdf_files)}] {pdf_path.name} | "
                    f"失败 | {elapsed:.1f}s | {record['error']}",
                    flush=True,
                )
            else:
                multimodal = record["result"]["multimodal"]
                print(
                    f"[{index}/{len(pdf_files)}] {pdf_path.name} | "
                    f"完成 ({status}) | {elapsed:.1f}s | "
                    f"多模态 processed={multimodal['processed']} "
                    f"failed={multimodal['failed']}",
                    flush=True,
                )
    finally:
        await rag.finalize_storages()

    print(
        "Batch index complete: "
        f"success={succeeded} partial={partial} failed={failed} skipped={skipped}"
    )
    print(f"Status: {status_file}")


if __name__ == "__main__":
    asyncio.run(main())
