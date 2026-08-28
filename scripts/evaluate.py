from __future__ import annotations

import argparse
import asyncio
import csv
import html
import json
import os
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from index import _build_rag, _configure_logging, _required_env
from lightrag.llm.openai import openai_complete_if_cache

from complex_rag.parser import MineruParser, Parser
from complex_rag.utils import (
    format_table_body,
    get_table_body,
    normalize_caption_list,
)


JUDGE_SYSTEM_PROMPT = """你是严格的金融问答评测员。只能根据给出的检索证据评分，不得使用外部知识。
请只返回一个 JSON 对象，不要输出 Markdown。"""

JUDGE_PROMPT = """请评估以下回答：

问题：
{question}

待评回答：
{answer}

实际检索证据：
{evidence}

分别给出 0 到 4 的整数分：
- faithfulness：回答中的事实性陈述是否得到检索证据支持。4=全部支持；3=仅有轻微未支持细节；2=支持与未支持并存；1=大部分未支持；0=完全不支持或与证据矛盾。
- answer_relevancy：回答是否直接、完整地回应问题。4=直接且完整；3=基本回应但略有遗漏或冗余；2=只回应一部分；1=几乎没有回应；0=无关。

返回格式：
{{
  "faithfulness": {{"score": 0, "reason": "简短理由", "unsupported_claims": []}},
  "answer_relevancy": {{"score": 0, "reason": "简短理由"}}
}}"""

CORRECTNESS_JUDGE_SYSTEM_PROMPT = """你是严格的金融问答正确性评测员。只能比较标准答案与待评回答，不得使用外部知识。
请只返回一个 JSON 对象，不要输出 Markdown。"""

CORRECTNESS_JUDGE_PROMPT = """请评估以下回答与标准答案是否一致：

问题：
{question}

标准答案：
{reference_answer}

待评回答：
{answer}

给出 answer_correctness 的 0 到 4 整数分：
- 4：核心结论完全正确，事实、数字、正负方向、单位和时期均与标准答案一致；数学等价表达或合理四舍五入也视为正确。
- 3：核心结论正确，仅有不影响结论的轻微精度、措辞或次要遗漏。
- 2：部分内容正确，但存在重要遗漏，或部分关键数字、单位、时期错误。
- 1：仅有少量信息正确，核心结论错误。
- 0：完全错误、与标准答案矛盾或没有回答问题。

不要因为表述方式不同而扣分；重点检查答案语义是否与标准答案一致。

返回格式：
{{
  "answer_correctness": {{"score": 0, "reason": "简短理由", "errors": []}}
}}"""

NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?%?")
EVALUATION_SCHEMA_VERSION = 4


def _load_queries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("queries file must contain a JSON array")
    return data


def _parse_pages(value: Any) -> list[int]:
    if isinstance(value, list):
        raw_pages = value
    else:
        raw_pages = str(value).split(",")
    pages = sorted({int(page) for page in raw_pages if str(page).strip()})
    if not pages or any(page < 1 for page in pages):
        raise ValueError(f"Invalid from_pages value: {value}")
    return pages


def _match_pdf(query_id: str, pdf_files: list[Path]) -> Path:
    matches = [path for path in pdf_files if query_id.startswith(f"{path.stem}_")]
    if not matches:
        raise ValueError(
            f"query-id does not match a PDF basename: {query_id}"
        )
    longest_length = max(len(path.stem) for path in matches)
    longest = [path for path in matches if len(path.stem) == longest_length]
    if len(longest) != 1:
        raise ValueError(f"query-id ambiguously matches PDF basenames: {query_id}")
    return longest[0]


def _load_content_list(
    output_root: Path,
    pdf_path: Path,
) -> list[dict[str, Any]]:
    """按索引阶段相同的 MinerU 目录规则读取并规范化解析结果。"""

    document_output = Parser._unique_output_dir(output_root, pdf_path)
    file_stem = pdf_path.stem
    if MineruParser._is_mineru_unsafe_windows_path(pdf_path):
        path_hash = MineruParser._mineru_safe_path_hash(pdf_path)
        file_stem = f"input_{path_hash}"
    content_list, _ = MineruParser._read_output_files(
        document_output,
        file_stem,
        method=os.getenv("PARSE_METHOD", "auto"),
    )
    if not content_list:
        raise FileNotFoundError(
            f"MinerU content_list not found or empty under: {document_output}"
        )
    return content_list


def _block_content(item: dict[str, Any]) -> str:
    content_type = str(item.get("type", "text"))
    if content_type == "text":
        return str(item.get("text", "")).strip()
    if content_type == "table":
        captions = normalize_caption_list(item.get("table_caption"))
        footnotes = normalize_caption_list(item.get("table_footnote"))
        image_path = str(item.get("img_path", "")).strip()
        parts = captions + [format_table_body(get_table_body(item))] + footnotes
        if image_path:
            parts.append(f"图片路径：{image_path}")
        return "\n".join(part for part in parts if part).strip()
    if content_type in {"image", "chart"}:
        captions = normalize_caption_list(
            item.get("image_caption", item.get("img_caption"))
        )
        footnotes = normalize_caption_list(
            item.get("image_footnote", item.get("img_footnote"))
        )
        image_path = str(item.get("img_path", "")).strip()
        parts = captions + footnotes
        if image_path:
            parts.append(f"图片路径：{image_path}")
        return "\n".join(parts).strip()
    return str(item.get("text", "")).strip()


def _gold_evidence(
    content_list: list[dict[str, Any]],
    pages: list[int],
) -> list[dict[str, Any]]:
    gold_pages = set(pages)
    evidence: list[dict[str, Any]] = []
    for index, item in enumerate(content_list):
        page_idx = int(item.get("page_idx", 0))
        if page_idx + 1 not in gold_pages:
            continue
        content = _block_content(item)
        if content:
            evidence.append(
                {
                    "content_list_index": index,
                    "type": item.get("type", "text"),
                    "page_idx": page_idx,
                    "page_number": page_idx + 1,
                    "content": content,
                }
            )
    return evidence


def _normalize_number_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
    normalized = normalized.replace("−", "-").replace("％", "%")
    return re.sub(r"\s+", "", normalized)


def _extract_numbers(text: str) -> list[str]:
    return NUMBER_PATTERN.findall(_normalize_number_text(text))


def _numeric_metrics(reference: str, answer: str) -> dict[str, Any]:
    reference_numbers = _extract_numbers(reference)
    normalized_answer = _normalize_number_text(answer)
    answer_numbers = _extract_numbers(normalized_answer)
    remaining_matches: dict[str, int] = {}

    def match_once(number: str) -> bool:
        pattern = rf"(?<![\d.]){re.escape(number)}(?![\d.])"
        if number not in remaining_matches:
            remaining_matches[number] = len(re.findall(pattern, normalized_answer))
        if remaining_matches[number] == 0:
            return False
        remaining_matches[number] -= 1
        return True

    matched = [number for number in reference_numbers if match_once(number)]
    applicable = bool(reference_numbers)
    return {
        "applicable": applicable,
        "reference_numbers": reference_numbers,
        "answer_numbers": sorted(set(answer_numbers)),
        "matched_numbers": matched,
        "accuracy": (
            len(matched) / len(reference_numbers) if reference_numbers else None
        ),
    }


def _hit_at_k(evidence: list[dict[str, Any]], pages: list[int], k: int) -> int:
    gold_pages = set(pages)
    retrieved_pages = {
        item.get("page_number") for item in evidence[:k] if item.get("page_number")
    }
    return int(bool(gold_pages & retrieved_pages))


def _question_types(query: dict[str, Any], pages: list[int]) -> dict[str, str]:
    category = str(query.get("category", "unknown"))
    modality, separator, task = category.partition("-")
    return {
        "category": category,
        "answer_type": str(query.get("answer_type", "unknown")),
        "modality": modality if separator else "unknown",
        "task": task if separator else category,
        "page_scope": "multi_page" if len(pages) > 1 else "single_page",
    }


def _build_judge() -> tuple[str, Callable[..., Any]]:
    api_key = _required_env("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or None
    model = os.getenv("JUDGE_MODEL", "").strip() or _required_env("LLM_MODEL")

    def judge_model_func(prompt: str, system_prompt: str | None = None):
        return openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=[],
            api_key=api_key,
            base_url=base_url,
        )

    return model, judge_model_func


def _parse_judge_response(response: str) -> dict[str, Any]:
    cleaned = re.sub(
        r"<think(?:ing)?>.*?</think(?:ing)?>",
        "",
        str(response),
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Judge did not return a JSON object")
    data = json.loads(cleaned[start : end + 1])
    result: dict[str, Any] = {}
    for name in ("faithfulness", "answer_relevancy"):
        metric = data[name]
        score = int(metric["score"])
        if not 0 <= score <= 4:
            raise ValueError(f"Judge score out of range for {name}: {score}")
        result[name] = {
            **metric,
            "score": score,
            "normalized_score": score / 4,
        }
    return result


def _parse_correctness_judge_response(response: str) -> dict[str, Any]:
    """解析独立正确性Judge返回的单个指标。"""

    cleaned = re.sub(
        r"<think(?:ing)?>.*?</think(?:ing)?>",
        "",
        str(response),
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Correctness Judge did not return a JSON object")
    data = json.loads(cleaned[start : end + 1])
    metric = data["answer_correctness"]
    score = int(metric["score"])
    if not 0 <= score <= 4:
        raise ValueError("Correctness Judge score out of range")
    return {
        **metric,
        "score": score,
        "normalized_score": score / 4,
    }


def _format_evidence_for_judge(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "（无检索证据）"
    parts = []
    for item in evidence:
        parts.append(
            f"[证据 {item['rank']} | 来源 {item['file_path']} | "
            f"页码 {item.get('page_number')}]\n{item['content']}"
        )
    return "\n\n".join(parts)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _latest_records(path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        query_id = str(record.get("query_id", ""))
        if query_id:
            latest[query_id] = record
    return list(latest.values())


def _validate_resume_schema(records: list[dict[str, Any]]) -> None:
    incompatible = [
        record.get("query_id", "")
        for record in records
        if record.get("evaluation_schema_version") != EVALUATION_SCHEMA_VERSION
    ]
    if incompatible:
        raise ValueError(
            "Existing results use an incompatible evaluation schema; "
            "use a new --result-dir"
        )


def _mean(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        successful = [record for record in group if record.get("status") != "error"]
        numeric_records = [
            item for item in successful if item.get("numeric", {}).get("applicable")
        ]
        judged_records = [item for item in successful if item.get("judge")]
        correctness_records = [
            item for item in successful if item.get("answer_correctness")
        ]
        return {
            "count": len(group),
            "completed": len(successful),
            "error_rate": (len(group) - len(successful)) / len(group),
            "hit_at_5": _mean([item.get("hit_at_5") for item in successful]),
            "numeric_count": len(numeric_records),
            "numeric_accuracy": _mean(
                [item.get("numeric", {}).get("accuracy") for item in numeric_records]
            ),
            "judge_count": len(judged_records),
            "faithfulness": _mean(
                [
                    item.get("judge", {})
                    .get("faithfulness", {})
                    .get("normalized_score")
                    for item in successful
                ]
            ),
            "answer_relevancy": _mean(
                [
                    item.get("judge", {})
                    .get("answer_relevancy", {})
                    .get("normalized_score")
                    for item in successful
                ]
            ),
            "correctness_judge_count": len(correctness_records),
            "answer_correctness": _mean(
                [
                    item.get("answer_correctness", {}).get("normalized_score")
                    for item in successful
                ]
            ),
            "query_seconds": _mean(
                [item.get("timing", {}).get("query_seconds") for item in successful]
            ),
            "judge_seconds": _mean(
                [item.get("timing", {}).get("judge_seconds") for item in successful]
            ),
            "correctness_judge_seconds": _mean(
                [
                    item.get("timing", {}).get("correctness_judge_seconds")
                    for item in successful
                ]
            ),
        }

    summary: dict[str, Any] = {"overall": summarize_group(records), "groups": {}}
    dimensions = {
        "category": lambda item: item.get("types", {}).get("category"),
        "answer_type": lambda item: item.get("types", {}).get("answer_type"),
        "page_scope": lambda item: item.get("types", {}).get("page_scope"),
        "modality": lambda item: item.get("types", {}).get("modality"),
        "document": lambda item: item.get("document"),
    }
    for dimension, getter in dimensions.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(getter(record) or "unknown")].append(record)
        summary["groups"][dimension] = {
            name: summarize_group(group) for name, group in sorted(grouped.items())
        }
    return summary


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    fieldnames = [
        "dimension",
        "value",
        "count",
        "completed",
        "error_rate",
        "hit_at_5",
        "numeric_count",
        "numeric_accuracy",
        "judge_count",
        "faithfulness",
        "answer_relevancy",
        "correctness_judge_count",
        "answer_correctness",
        "query_seconds",
        "judge_seconds",
        "correctness_judge_seconds",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"dimension": "overall", "value": "all", **summary["overall"]})
        for dimension, groups in summary["groups"].items():
            for value, metrics in groups.items():
                writer.writerow(
                    {"dimension": dimension, "value": value, **metrics}
                )


def _evidence_html(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "<p>无证据</p>"
    blocks = []
    for index, item in enumerate(evidence, start=1):
        page = item.get("page_number")
        kind = item.get("type", item.get("original_type", "text"))
        content = html.escape(str(item.get("content", "")))
        blocks.append(
            f"<details><summary>#{index} · page {page} · {html.escape(str(kind))}</summary>"
            f"<pre>{content}</pre></details>"
        )
    return "".join(blocks)


def _write_html_report(
    path: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    cards = []
    for record in records:
        if record.get("status") == "error":
            cards.append(
                f"<section><h2>{html.escape(record.get('query_id', ''))}</h2>"
                f"<p class='error'>{html.escape(record.get('error', ''))}</p></section>"
            )
            continue
        types = record["types"]
        numeric_accuracy = record.get("numeric", {}).get("accuracy")
        faithfulness = (
            record.get("judge", {}).get("faithfulness", {}).get("normalized_score")
        )
        relevancy = (
            record.get("judge", {})
            .get("answer_relevancy", {})
            .get("normalized_score")
        )
        correctness = record.get("answer_correctness", {}).get(
            "normalized_score"
        )
        judge_reason = "；".join(
            str(metric.get("reason", ""))
            for metric in record.get("judge", {}).values()
            if metric.get("reason")
        )
        correctness_reason = str(
            record.get("answer_correctness", {}).get("reason", "")
        )
        cards.append(
            "<section>"
            f"<h2>{html.escape(record['query_id'])}</h2>"
            f"<p>{html.escape(record['question'])}</p>"
            f"<p class='meta'>{html.escape(types['category'])} · "
            f"{html.escape(types['answer_type'])} · gold pages "
            f"{html.escape(str(record['from_pages']))} · "
            f"Hit@5={record['hit_at_5']} · "
            f"Numeric={numeric_accuracy} · Faithfulness={faithfulness} · "
            f"Relevancy={relevancy} · Correctness={correctness}</p>"
            f"<p class='meta'>证据Judge：{html.escape(judge_reason)}</p>"
            f"<p class='meta'>正确性Judge：{html.escape(correctness_reason)}</p>"
            "<div class='grid'>"
            f"<div><h3>标准答案</h3><pre>{html.escape(record['reference_answer'])}</pre></div>"
            f"<div><h3>生成答案</h3><pre>{html.escape(record['generated_answer'])}</pre></div>"
            f"<div><h3>标准页证据</h3>{_evidence_html(record['gold_evidence'])}</div>"
            f"<div><h3>检索证据</h3>{_evidence_html(record['retrieved_evidence'])}</div>"
            "</div></section>"
        )
    overall = html.escape(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>RAG 评测报告</title>
<style>
body{{font-family:Arial,"Microsoft YaHei",sans-serif;margin:24px;color:#222}}
section{{border:1px solid #ddd;border-radius:8px;padding:16px;margin:18px 0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.grid>div{{min-width:0;background:#fafafa;padding:12px;border-radius:6px}}
pre{{white-space:pre-wrap;word-break:break-word;font-family:inherit}}
.meta{{color:#666}}.error{{color:#a00}}summary{{cursor:pointer;margin:8px 0}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><h1>RAG 评测报告</h1><h2>总体指标</h2><pre>{overall}</pre>
{''.join(cards)}</body></html>"""
    path.write_text(document, encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Chinese PDF RAG evaluation")
    parser.add_argument("--queries", default="data/queries_ch.json")
    parser.add_argument("--pdf-dir", default="data/pdf_ch")
    parser.add_argument("--output-root", default=os.getenv("OUTPUT_DIR", "./output"))
    parser.add_argument("--result-dir", default="experiment_results/eval_ch")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _configure_logging(args.verbose)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be a positive integer")
    queries_path = Path(args.queries).resolve()
    pdf_dir = Path(args.pdf_dir).resolve()
    output_root = Path(args.output_root).resolve()
    result_dir = Path(args.result_dir).resolve()
    if not queries_path.is_file():
        raise FileNotFoundError(f"Queries file does not exist: {queries_path}")
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")
    if not output_root.is_dir():
        raise FileNotFoundError(
            f"MinerU output directory does not exist: {output_root}"
        )
    result_file = result_dir / "results.jsonl"
    if result_file.exists() and not args.resume:
        raise FileExistsError(
            f"Result file already exists: {result_file}; use --resume or another --result-dir"
        )

    queries = _load_queries(queries_path)
    if args.limit is not None:
        queries = queries[: args.limit]
    pdf_files = sorted(pdf_dir.glob("*.pdf"), key=lambda path: path.name)
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {pdf_dir}")
    existing_records = _latest_records(result_file) if result_file.exists() else []
    if args.resume:
        _validate_resume_schema(existing_records)
    completed = {
        record["query_id"]
        for record in existing_records
        if record.get("status") == "success"
    }
    judge_model, judge_model_func = _build_judge()
    rag = _build_rag()
    content_cache: dict[Path, list[dict[str, Any]]] = {}

    try:
        for index, query in enumerate(queries, start=1):
            query_id = str(query["query-id"])
            if query_id in completed:
                print(f"[{index}/{len(queries)}] skipped {query_id}")
                continue

            started_at = time.perf_counter()
            try:
                pages = _parse_pages(query["from_pages"])
                pdf_path = _match_pdf(query_id, pdf_files)
                if pdf_path not in content_cache:
                    content_cache[pdf_path] = _load_content_list(
                        output_root,
                        pdf_path,
                    )
                gold = _gold_evidence(content_cache[pdf_path], pages)

                query_started = time.perf_counter()
                trace = await rag.aquery_with_trace(
                    str(query["query"]),
                    mode="mix",
                    response_type="简洁的最终答案",
                )
                query_seconds = time.perf_counter() - query_started
                answer = trace["answer"]
                evidence = trace["evidence"]
                if any(item.get("page_number") is None for item in evidence):
                    raise RuntimeError(
                        "Retrieved evidence contains chunks without page_idx; "
                        "rebuild the index in a new WORKING_DIR"
                    )

                judge_started = time.perf_counter()
                judge_error = None
                judge: dict[str, Any] = {}
                try:
                    judge_prompt = JUDGE_PROMPT.format(
                        question=query["query"],
                        answer=answer,
                        evidence=_format_evidence_for_judge(evidence),
                    )
                    judge_response = await judge_model_func(
                        judge_prompt,
                        system_prompt=JUDGE_SYSTEM_PROMPT,
                    )
                    judge = _parse_judge_response(str(judge_response))
                except Exception as exc:
                    judge_error = str(exc)
                judge_seconds = time.perf_counter() - judge_started

                correctness_judge_started = time.perf_counter()
                correctness_judge_error = None
                answer_correctness: dict[str, Any] = {}
                try:
                    correctness_prompt = CORRECTNESS_JUDGE_PROMPT.format(
                        question=query["query"],
                        reference_answer=query["answer"],
                        answer=answer,
                    )
                    correctness_response = await judge_model_func(
                        correctness_prompt,
                        system_prompt=CORRECTNESS_JUDGE_SYSTEM_PROMPT,
                    )
                    answer_correctness = _parse_correctness_judge_response(
                        str(correctness_response)
                    )
                except Exception as exc:
                    correctness_judge_error = str(exc)
                correctness_judge_seconds = (
                    time.perf_counter() - correctness_judge_started
                )

                record = {
                    "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
                    "query_id": query_id,
                    "status": (
                        "success"
                        if judge_error is None and correctness_judge_error is None
                        else "partial"
                    ),
                    "document": pdf_path.name,
                    "question": query["query"],
                    "types": _question_types(query, pages),
                    "from_pages": pages,
                    "reference_answer": str(query["answer"]),
                    "generated_answer": answer,
                    "gold_evidence": gold,
                    "retrieved_evidence": evidence,
                    "hit_at_5": _hit_at_k(evidence, pages, 5),
                    "numeric": _numeric_metrics(str(query["answer"]), answer),
                    "judge_model": judge_model,
                    "judge": judge,
                    "judge_error": judge_error,
                    "answer_correctness": answer_correctness,
                    "correctness_judge_error": correctness_judge_error,
                    "timing": {
                        "query_seconds": round(query_seconds, 3),
                        "judge_seconds": round(judge_seconds, 3),
                        "correctness_judge_seconds": round(
                            correctness_judge_seconds,
                            3,
                        ),
                        "total_seconds": round(time.perf_counter() - started_at, 3),
                    },
                }
            except Exception as exc:
                record = {
                    "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
                    "query_id": query_id,
                    "status": "error",
                    "question": query.get("query", ""),
                    "error": str(exc),
                    "timing": {
                        "total_seconds": round(time.perf_counter() - started_at, 3)
                    },
                }

            _append_jsonl(result_file, record)
            print(f"[{index}/{len(queries)}] {record['status']} {query_id}")
    finally:
        await rag.finalize_storages()

    records = _latest_records(result_file)
    summary = _summarize(records)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary_csv(result_dir / "summary.csv", summary)
    _write_html_report(result_dir / "report.html", records, summary)
    print(f"Results: {result_file}")
    print(f"Summary: {result_dir / 'summary.csv'}")
    print(f"Report: {result_dir / 'report.html'}")


if __name__ == "__main__":
    asyncio.run(main())
