# complex_rag_demo

`complex_rag_demo` 是面向包含正文、表格、图片和图表的中文金融 PDF 的多模态 RAG 问答系统。
项目使用 MinerU 解析文档，使用 LightRAG 保存文本块、向量和知识图谱，并根据实际
召回内容选择文本模型或视觉模型回答。

仓库只发布源码和配置模板，不包含 PDF、MinerU 解析产物、LightRAG storage、
评测数据或模型凭据。首次使用必须在本机重新解析文档并建立索引。

## 1. 快速理解项目

### 1.1 离线索引

```text
PDF
  → MinerU解析
  → content_list（text/table/image/chart）
  → 正文与多模态条目分流
      ├─ text → 按页聚合 → LightRAG.ainsert()
      │          → 页内token分块并保留page_idx
      │          → chunk向量化
      │          → 实体关系抽取
      │          → 图谱去重合并
      │
      └─ table/image/chart
                 → 收集邻近正文
                 → LLM/VLM生成描述与多模态主实体
                 → 构造完整多模态chunk
                 → chunk和主实体入库
                 → LightRAG再次抽取细粒度实体关系
                 → 添加belongs_to关系
                 → 图谱去重合并
```

### 1.2 在线查询

```text
用户问题
  → LightRAG mix宽召回30条候选chunk
  → 外部reranker按问题与chunk相关性重新打分
  → 保留最终Top 5
  → 得到包含Top 5检索上下文和用户问题的完整Prompt
  → 检查Prompt中的图片路径
      ├─ 有合法图片 → Prompt + 原图 → VLM回答
      └─ 无合法图片 → 同一Prompt → 文本LLM回答
```


## 2. 首次使用：重新解析并建立 storage

运行前编辑本地 `.env`，至少填写 `OPENAI_API_KEY`、`LLM_MODEL`、`VISION_MODEL` 和
`EMBEDDING_MODEL`。如果使用兼容 OpenAI API 的服务，再设置 `OPENAI_BASE_URL`。
embedding 输出维度如果不是默认的 1536，需要在 `.env` 中增加正确的
`EMBEDDING_DIM`；维度变化后必须换用新的空 storage。

仓库不提供 PDF、MinerU `output/` 或 LightRAG `rag_storage/`。首次使用先创建本地
数据目录，将有权使用的 PDF 放入对应目录，然后运行下面的索引命令。首次索引必须使用一个
不存在或为空的 `WORKING_DIR`。不要复制他人的 storage，也不要在旧版、不同
embedding 维度或不同解析结果生成的 storage 上追加。

MinerU 图片会保存在 `OUTPUT_DIR`。VLM 查询依赖这些图片的实际本地路径，因此完成
索引后仍需保留对应的解析产物；移动项目、清理 `output/` 或复制 storage 到另一台机器
都可能导致历史图片无法参与查询。

## 3. 运行命令

### 3.1 索引单个 PDF
示例：
```powershell
.\.venv\Scripts\python.exe -B scripts\index.py data\pdf\example.pdf --lang ch
```

### 3.2 批量索引
示例：
```powershell
.\.venv\Scripts\python.exe -B scripts\batch_index.py data\pdf_ch --lang ch
```

脚本按文件名顺序处理 PDF，默认状态文件为
`experiment_results/index_status.jsonl`。首次运行会拒绝非空 `WORKING_DIR`。中断后用
相同命令追加 `--resume`：

### 3.3 正式评测
示例：
```powershell
python -B scripts\evaluate.py --queries data\queries_ch.json --pdf-dir data\pdf_ch --result-dir experiment_results/eval_ch_v1
```

评测只使用 `mix`，统一宽召回30条候选chunk，经reranker重排后输出最终Top 5，
并计算页级 Hit@5 和数字准确性。生成质量通过两次相互隔离的Judge调用评估：

- 第一次只读取问题、生成答案和实际Top 5证据，计算 Faithfulness 和 Answer Relevancy。
- 第二次只读取问题、标准答案和生成答案，计算 Answer Correctness，避免标准答案污染Faithfulness。

## 4. 致谢与参考

本项目的整体架构和部分实现思路参考了香港大学 Data Intelligence Lab 开源的
[RAG-Anything](https://github.com/HKUDS/RAG-Anything)，包括 `RAGAnything` 核心对象、
MinerU 内容列表的多模态分流、模态处理器以及检索后按图片召回情况选择 VLM 的设计。
RAG-Anything 采用 MIT License；本仓库在 [LICENSE](LICENSE) 中保留了相应的上游
版权声明。

本项目是在该思路上针对中文金融 PDF demo 所做的独立实现和调整，主要增加或强化了
按页分块与 `page_idx`、多模态主实体及 `belongs_to` 关系、Windows MinerU 路径兼容、
带实际检索证据的评测流程，以及批量实验的状态记录与断点续跑。本项目并非 RAG-Anything
官方发行版，也不代表上游项目或其作者提供背书。

如果在研究或论文中使用本项目，请同时引用 RAG-Anything 官方论文：

```bibtex
@misc{guo2025raganythingallinoneragframework,
  title={RAG-Anything: All-in-One RAG Framework},
  author={Zirui Guo and Xubin Ren and Lingrui Xu and Jiahao Zhang and Chao Huang},
  year={2025},
  eprint={2510.12323},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2510.12323}
}
```

相关基础项目：

- [LightRAG](https://github.com/HKUDS/LightRAG)：知识图谱与向量检索基础。
- [MinerU](https://github.com/opendatalab/MinerU)：PDF 解析与多模态内容提取。
