"""为回答阶段提供不执行任意代码的十进制计算器。"""

from __future__ import annotations

import ast
import re
from decimal import Decimal, DecimalException, ROUND_HALF_UP, localcontext
from typing import Final

from lightrag.utils import logger


CALCULATION_MARKER: Final[re.Pattern[str]] = re.compile(
    r"\[\[CALC:(.*?)\]\]",
    flags=re.DOTALL,
)
ALLOWED_EXPRESSION: Final[re.Pattern[str]] = re.compile(r"[0-9eE._+\-*/()\s]+")
MAX_EXPRESSION_LENGTH = 256
MAX_DECIMAL_PLACES = 12
MAX_MAGNITUDE_EXPONENT = 100


class SafeCalculationError(ValueError):
    """表示表达式不符合安全计算器约束。"""


def evaluate_safe_expression(expression: str) -> Decimal:
    """仅使用 Decimal 计算数字、括号和四则运算。"""

    normalized = _normalize_expression(expression)
    if (
        not normalized
        or len(normalized) > MAX_EXPRESSION_LENGTH
        or ALLOWED_EXPRESSION.fullmatch(normalized) is None
    ):
        raise SafeCalculationError("计算表达式为空、过长或包含非法字符")

    try:
        parsed = ast.parse(normalized, mode="eval")
        with localcontext() as context:
            context.prec = 50
            result = _evaluate_node(parsed.body, normalized)
    except (SyntaxError, DecimalException, RecursionError) as exc:
        raise SafeCalculationError("计算表达式无效") from exc
    return _check_decimal(result)


def replace_calculation_markers(answer: str) -> str:
    """计算答案中的内部占位符并替换为最终数字。"""

    def replace_marker(match: re.Match[str]) -> str:
        """计算单个占位符，失败时返回明确提示。"""

        try:
            expression, decimal_places = _split_marker_body(match.group(1))
            result = evaluate_safe_expression(expression)
            return _format_decimal(result, decimal_places)
        except SafeCalculationError as exc:
            logger.warning("Safe calculation failed: %s", exc)
            return "计算失败"

    return CALCULATION_MARKER.sub(replace_marker, answer)


def _normalize_expression(expression: str) -> str:
    """统一模型可能输出的全角括号、运算符和千分位。"""

    normalized = expression.strip().translate(
        str.maketrans(
            {
                "（": "(",
                "）": ")",
                "＋": "+",
                "－": "-",
                "−": "-",
                "–": "-",
                "—": "-",
                "×": "*",
                "÷": "/",
            }
        )
    )
    thousands_separator = re.compile(r"(?<=\d)[,，](?=\d{3}(?:\D|$))")
    while thousands_separator.search(normalized):
        normalized = thousands_separator.sub("", normalized)
    return normalized


def _evaluate_node(node: ast.AST, expression: str) -> Decimal:
    """递归解释白名单 AST 节点，拒绝其他 Python 语法。"""

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise SafeCalculationError("表达式只能包含数字")
        source = ast.get_source_segment(expression, node)
        if source is None:
            raise SafeCalculationError("无法读取数字")
        return _check_decimal(Decimal(source.replace("_", "")))

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_node(node.operand, expression)
        return _check_decimal(value if isinstance(node.op, ast.UAdd) else -value)

    if not isinstance(node, ast.BinOp):
        raise SafeCalculationError("表达式包含不允许的语法")

    left = _evaluate_node(node.left, expression)
    right = _evaluate_node(node.right, expression)
    if isinstance(node.op, ast.Add):
        return _check_decimal(left + right)
    if isinstance(node.op, ast.Sub):
        return _check_decimal(left - right)
    if isinstance(node.op, ast.Mult):
        return _check_decimal(left * right)
    if isinstance(node.op, ast.Div):
        if right == 0:
            raise SafeCalculationError("除数不能为零")
        return _check_decimal(left / right)
    raise SafeCalculationError("只允许四则运算")


def _check_decimal(value: Decimal) -> Decimal:
    """拒绝非有限值和异常大的结果，限制资源消耗。"""

    if not value.is_finite():
        raise SafeCalculationError("计算结果不是有限数字")
    if value and abs(value.adjusted()) > MAX_MAGNITUDE_EXPONENT:
        raise SafeCalculationError("计算结果数量级过大")
    return value


def _split_marker_body(marker_body: str) -> tuple[str, int | None]:
    """从占位符中拆出表达式和可选的小数位数。"""

    expression, separator, precision_text = marker_body.rpartition("|")
    if not separator:
        return marker_body.strip(), None
    precision_text = precision_text.strip()
    if not precision_text.isdigit():
        return marker_body.strip(), None
    decimal_places = int(precision_text)
    if decimal_places > MAX_DECIMAL_PLACES:
        raise SafeCalculationError("小数位数超过限制")
    return expression.strip(), decimal_places


def _format_decimal(value: Decimal, decimal_places: int | None) -> str:
    """按模型指定精度四舍五入并输出普通十进制文本。"""

    if decimal_places is not None:
        quantizer = Decimal(1).scaleb(-decimal_places)
        with localcontext() as context:
            context.prec = 128
            value = value.quantize(quantizer, rounding=ROUND_HALF_UP)
        if value == 0:
            value = abs(value)
        result = format(value, f".{decimal_places}f")
    else:
        result = format(value, "f")
        if "." in result:
            result = result.rstrip("0").rstrip(".")
    return "0" if result == "-0" else result
