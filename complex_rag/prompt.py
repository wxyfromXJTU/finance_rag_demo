"""图片、表格处理和 VLM 增强查询使用的提示词。"""

# system prompt 只规定模型角色，具体任务和输出格式放在 analysis prompt 中。
IMAGE_ANALYSIS_SYSTEM = "你是一位专业的图像分析专家。请提供详细、准确的描述。"
TABLE_ANALYSIS_SYSTEM = (
    "你是一位专业的数据分析师。请提供包含具体数值和洞察的表格分析。"
)
VLM_QUERY_SYSTEM = (
    "你是一位中文金融文档问答助手。请严格依据提供的 Top 5 检索上下文和图片回答，"
    "不要补充证据之外的信息。表格题先核对指标行、时期列和单位；图表题先核对标题、"
    "图例、横轴时期或类别、纵轴单位，避免混用相邻图表。比较、极值或计算题只提取问题"
    "必需的数据。需要计算时必须将最终结果写成[[CALC:四则运算表达式|小数位数]]，"
    "由安全计算器执行；不要自行填写计算结果。最终只给简洁结论，必要时最多附一行"
    "取数依据或计算式；"
    "不输出 JSON、冗长推理或参考文献列表，证据不足时明确说明。"
)


CONCISE_ANSWER_PROMPT = """请回答下面的用户问题：
{query}

回答要求：
- 先判断问题需要直接取数、日期定位、趋势归纳、比较、极值还是计算，不要加入无关数字。
- 表格取数必须认真理解题意，匹配行与列，防止取错数字；
- 涉及图表的问题必须根据题意更细致地分析图表，匹配标题、图例、横轴和纵轴，不能只靠图表的文字描述。
- 最高、最低、平均值及增长率等问题应检查完整相关区间，不能拿单个数据点代替。
- 需要计算时，把答案中的每个计算结果写成 [[CALC:四则运算表达式|小数位数]]，例如增长率为 [[CALC:(120-100)/100*100|2]]%；不要自行填写计算结果。
- CALC 表达式只能包含阿拉伯数字、括号和 + - * /，不写单位、千分位、等号或变量。
- 必要时可在答案后用一行给出原始数值和同一个 CALC 计算式。
- 直接给出简洁自然语言答案，不输出 JSON、分析步骤或参考文献列表。
- 证据不足或图片无法辨认时明确说明，不要猜测。"""


IMAGE_ANALYSIS_PROMPT = """请结合文档信息分析这张图片，并只返回以下 JSON：
{{
  "detailed_description": "适合知识检索的详细描述",
  "entity_info": {{
    "entity_name": "{entity_name}",
    "entity_type": "image",
    "summary": "不超过100字的摘要"
  }}
}}

文档信息：
- 周围正文：{context}
- 章节路径：{section_path}
- 图片路径：{image_path}
- 标题：{captions}
- 脚注：{footnotes}

要求：
- 描述图片中的文字、对象、结构及其关系。
- 如果图片是图表，说明指标、具体数值、趋势和比较关系。
- 实体名称应有明确语义，不要使用文件名或无实际含义的图号。"""


# 表格已经由 MinerU 转成结构化文本，因此直接交给文本模型分析。
TABLE_ANALYSIS_PROMPT = """请结合文档信息分析以下表格，并只返回以下 JSON：
{{
  "detailed_description": "包含表格结构、关键数值、趋势和比较关系的描述",
  "entity_info": {{
    "entity_name": "{entity_name}",
    "entity_type": "table",
    "summary": "不超过100字的摘要"
  }}
}}

文档信息：
- 周围正文：{context}
- 图片路径：{table_img_path}
- 标题：{table_caption}
- 表格内容：{table_body}
- 脚注：{table_footnote}

要求：
- 说明表头、数据结构和各字段含义。
- 保留关键指标和具体数值，并总结趋势与比较关系。
- 结合周围正文解释表格作用；没有上下文时仅依据表格内容分析。
- 实体名称应有明确语义，不要使用文件名或无实际含义的表号。"""


# 以下模板只负责把原始信息和模型描述整理成写入 LightRAG 的文本 chunk。
IMAGE_CHUNK_TEMPLATE = """图片内容分析：
章节路径：{section_path}
邻近文本：{neighbor_text}
图片路径：{image_path}
标题：{captions}
脚注：{footnotes}

视觉分析：{enhanced_caption}"""


TABLE_CHUNK_TEMPLATE = """表格分析：
图片路径：{table_img_path}
标题：{table_caption}
结构：{table_body}
脚注：{table_footnote}

分析：{enhanced_caption}"""
