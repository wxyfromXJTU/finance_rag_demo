from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


_IS_WINDOWS = platform.system() == "Windows"


class MineruExecutionError(RuntimeError):
    """MinerU 命令返回非零退出码。"""

    def __init__(self, return_code: int, error_msg: str) -> None:
        self.return_code = return_code
        self.error_msg = error_msg
        super().__init__(
            f"MinerU command failed with return code {return_code}: {error_msg}"
        )


class Parser:

    logger = logging.getLogger(__name__)

    @staticmethod
    def _unique_output_dir(base_dir: str | Path, file_path: str | Path) -> Path:
        """为文件生成稳定且不易冲突的解析输出目录。"""

        resolved_file = Path(file_path).resolve()
        path_hash = hashlib.md5(str(resolved_file).encode("utf-8")).hexdigest()[:8]
        return Path(base_dir) / f"{resolved_file.stem}_{path_hash}"

    def parse_pdf(
        self,
        pdf_path: str | Path,
        output_dir: str | None = None,
        method: str = "auto",
        lang: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """解析 PDF 并返回按文档顺序排列的内容块。"""

        raise NotImplementedError

    def check_installation(self) -> bool:
        """检查 Parser 的外部运行程序是否可用。"""

        raise NotImplementedError


class MineruParser(Parser):

    @classmethod
    def _is_mineru_unsafe_windows_path(cls, path: str | Path) -> bool:
        """判断路径是否需要先转移到临时 ASCII 目录。"""

        if not _IS_WINDOWS:
            return False

        parsed_path = Path(path)
        try:
            str(parsed_path).encode("ascii")
        except UnicodeEncodeError:
            return True

        return any(part.endswith((" ", ".")) for part in parsed_path.parts)

    @classmethod
    def _mineru_safe_path_hash(cls, path: str | Path) -> str:
        """生成临时路径使用的稳定短哈希。"""

        path_text = str(Path(path).resolve())
        return hashlib.md5(path_text.encode("utf-8")).hexdigest()[:10]

    @classmethod
    def _prepare_mineru_paths(
        cls,
        input_path: str | Path,
        output_dir: str | Path,
    ) -> tuple[Path, Path, str, Path | None]:
        """必要时为 MinerU 准备只含 ASCII 字符的临时输入输出路径。"""

        input_path = Path(input_path)
        output_dir = Path(output_dir)
        input_is_unsafe = cls._is_mineru_unsafe_windows_path(input_path)
        output_is_unsafe = cls._is_mineru_unsafe_windows_path(output_dir)

        if not input_is_unsafe and not output_is_unsafe:
            return input_path, output_dir, input_path.stem, None

        path_hash = cls._mineru_safe_path_hash(input_path)
        temp_dir = Path(tempfile.mkdtemp(prefix="complex_rag_mineru_"))

        mineru_input_path = input_path
        if input_is_unsafe:
            mineru_input_path = temp_dir / f"input_{path_hash}{input_path.suffix.lower()}"
            shutil.copy2(input_path, mineru_input_path)

        mineru_output_dir = output_dir
        if output_is_unsafe:
            mineru_output_dir = temp_dir / f"mineru_{path_hash}"
            mineru_output_dir.mkdir(parents=True, exist_ok=True)

        return mineru_input_path, mineru_output_dir, mineru_input_path.stem, temp_dir

    @classmethod
    def _copy_mineru_output_tree(cls, source_dir: Path, target_dir: Path) -> None:
        """把临时 MinerU 输出复制回用户指定目录。"""

        if source_dir == target_dir:
            return

        target_dir.mkdir(parents=True, exist_ok=True)
        if not source_dir.exists():
            return

        for item in source_dir.iterdir():
            target = target_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

    @classmethod
    def _cleanup_mineru_temp_dir(cls, temp_dir: Path | None) -> None:
        """删除为 MinerU 创建的临时目录。"""

        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    @classmethod
    def _run_mineru_command(
        cls,
        input_path: str | Path,
        output_dir: str | Path,
        method: str = "auto",
        lang: str | None = None,
        backend: str | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
        formula: bool = True,
        table: bool = True,
        device: str | None = None,
        vlm_url: str | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> None:
        """运行 MinerU CLI，并把失败信息转换为明确异常。"""

        command = [
            "mineru",
            "-p",
            str(input_path),
            "-o",
            str(output_dir),
            "-m",
            method,
        ]

        if backend:
            command.extend(["-b", backend])
        if lang:
            command.extend(["-l", lang])
        if start_page is not None:
            command.extend(["-s", str(start_page)])
        if end_page is not None:
            command.extend(["-e", str(end_page)])
        if not formula:
            command.extend(["-f", "false"])
        if not table:
            command.extend(["-t", "false"])
        if device:
            command.extend(["-d", device])
        if vlm_url:
            command.extend(["-u", vlm_url])

        custom_env = kwargs.pop("env", None)
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported MinerU arguments: {unsupported}")

        process_env = None
        if custom_env is not None:
            if not isinstance(custom_env, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in custom_env.items()
            ):
                raise TypeError("env must be a dictionary of string keys and values")
            process_env = os.environ.copy()
            process_env.update(custom_env)

        subprocess_kwargs: dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "ignore",
            "env": process_env,
            "timeout": timeout,
            "check": False,
        }
        if _IS_WINDOWS:
            subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        cls.logger.info("Executing MinerU command: %s", subprocess.list2cmdline(command))
        try:
            result = subprocess.run(command, **subprocess_kwargs)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "mineru command not found; install it with: pip install 'mineru[core]'"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"MinerU did not finish within {timeout}s") from exc

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise MineruExecutionError(result.returncode, error_msg)

        if result.stdout.strip():
            cls.logger.debug("[MinerU stdout] %s", result.stdout.strip())
        if result.stderr.strip():
            cls.logger.debug("[MinerU stderr] %s", result.stderr.strip())

    @classmethod
    def _read_output_files(
        cls,
        output_dir: Path,
        file_stem: str,
        method: str = "auto",
    ) -> tuple[list[dict[str, Any]], str]:
        """读取 MinerU 生成的 content_list、Markdown 并修正资源路径。"""

        markdown_file = output_dir / f"{file_stem}.md"
        content_list_file = output_dir / f"{file_stem}_content_list.json"
        images_base_dir = output_dir

        file_stem_dir = output_dir / file_stem
        if file_stem_dir.is_dir():
            found = False
            for subdir in file_stem_dir.iterdir():
                if not subdir.is_dir():
                    continue
                candidate = subdir / f"{file_stem}_content_list.json"
                if candidate.exists():
                    markdown_file = subdir / f"{file_stem}.md"
                    content_list_file = candidate
                    images_base_dir = subdir
                    found = True
                    break

            if not found:
                images_base_dir = file_stem_dir / method
                markdown_file = images_base_dir / f"{file_stem}.md"
                content_list_file = images_base_dir / f"{file_stem}_content_list.json"

        markdown = ""
        if markdown_file.exists():
            markdown = markdown_file.read_text(encoding="utf-8")

        if not content_list_file.exists():
            cls.logger.warning("MinerU content list not found: %s", content_list_file)
            return [], markdown

        try:
            content_list = json.loads(content_list_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            cls.logger.warning("Could not read MinerU content list: %s", exc)
            return [], markdown

        if not isinstance(content_list, list):
            cls.logger.warning("MinerU content list must be a JSON array")
            return [], markdown

        resolved_base = images_base_dir.resolve()
        for item in content_list:
            if not isinstance(item, dict):
                continue

            if "img_caption" in item and "image_caption" not in item:
                item["image_caption"] = item["img_caption"]
            elif "image_caption" in item and "img_caption" not in item:
                item["img_caption"] = item["image_caption"]

            if "img_footnote" in item and "image_footnote" not in item:
                item["image_footnote"] = item["img_footnote"]
            elif "image_footnote" in item and "img_footnote" not in item:
                item["img_footnote"] = item["image_footnote"]

            for field_name in ("img_path", "table_img_path", "equation_img_path"):
                raw_path = item.get(field_name)
                if not raw_path:
                    continue

                absolute_path = (images_base_dir / str(raw_path)).resolve()
                if not absolute_path.is_relative_to(resolved_base):
                    cls.logger.warning(
                        "Unsafe MinerU media path in %s: %s", field_name, raw_path
                    )
                    item[field_name] = ""
                    continue

                item[field_name] = str(absolute_path)

        return content_list, markdown

    def parse_pdf(
        self,
        pdf_path: str | Path,
        output_dir: str | None = None,
        method: str = "auto",
        lang: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """调用 MinerU 解析一个本地 PDF。"""

        source_path = Path(pdf_path)
        if not source_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {source_path}")
        if not source_path.is_file():
            raise ValueError(f"PDF path is not a file: {source_path}")

        if output_dir:
            base_output_dir = self._unique_output_dir(output_dir, source_path)
        else:
            base_output_dir = source_path.parent / "mineru_output"
        base_output_dir.mkdir(parents=True, exist_ok=True)

        mineru_input, mineru_output, file_stem, temp_dir = self._prepare_mineru_paths(
            source_path, base_output_dir
        )
        try:
            self._run_mineru_command(
                input_path=mineru_input,
                output_dir=mineru_output,
                method=method,
                lang=lang,
                **kwargs,
            )
            self._copy_mineru_output_tree(mineru_output, base_output_dir)

            backend = kwargs.get("backend") or ""
            output_method = method
            if backend.startswith("vlm-"):
                output_method = "vlm"
            elif backend.startswith("hybrid-"):
                output_method = "hybrid_auto"

            content_list, _ = self._read_output_files(
                base_output_dir,
                file_stem,
                method=output_method,
            )
            return content_list
        finally:
            self._cleanup_mineru_temp_dir(temp_dir)

    def check_installation(self) -> bool:
        """通过 ``mineru --version`` 检查 CLI 是否可执行。"""

        subprocess_kwargs: dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "check": True,
            "encoding": "utf-8",
            "errors": "ignore",
        }
        if _IS_WINDOWS:
            subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        try:
            result = subprocess.run(["mineru", "--version"], **subprocess_kwargs)
            self.logger.info("MinerU version: %s", result.stdout.strip())
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning(
                "MinerU is not available; install it with: pip install 'mineru[core]'"
            )
            return False
