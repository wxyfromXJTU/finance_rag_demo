"""图片、表格处理和 VLM 增强查询使用的提示词。"""

# system prompt 只规定模型角色，具体任务和输出格式放在 analysis prompt 中。
IMAGE_ANALYSIS_SYSTEM = "你是一位专业的图像分析专家。请提供详细、准确的描述。"
TABLE_ANALYSIS_SYSTEM = (
    "你是一位专业的数据分析师。请提供包含具体数值和洞察的表格分析。"
)
VLM_QUERY_SYSTEM = (
    "你是一位能够综合分析文本和图片的助手。"
    "请严格依据提供的检索上下文和图片回答问题；证据不足时明确说明。"
)


# 图片和图表共用一个分析模板；没有上下文时由调用方把“无”填入 context。
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
