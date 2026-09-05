# zsttSystem

面向高校培养方案与课程教学大纲的 RAG-KG 问答系统。系统使用：

- FastAPI 提供查询、课程和反馈 API。
- ChromaDB 保存课程文本向量并执行语义检索。
- 本地 Neo4j Community 保存课程依赖图并执行图查询。
- DeepSeek 或其他兼容 OpenAI Chat Completions 的模型生成基于证据的回答。

项目不依赖 LightRAG，也不需要单独的 Embedding HTTP 服务。

## 架构

```text
DOCX/XLSX
    │
    ▼
离线管线：parse → concept → graph → embed
    │                    │        │
    │                    │        └── ChromaDB
    │                    └─────────── Neo4j Community
    ▼
FastAPI → QueryRouter → ChromaDB / Neo4j → DeepSeek
```

## 运行要求

- Python 3.11 或 3.12
- Docker Desktop（用于启动 ChromaDB 和 Neo4j Community）

DeepSeek API 密钥是可选项。配置后，系统会使用兼容 OpenAI Chat
Completions 的模型生成基于证据的回答；未配置或生成服务不可用时，系统会
根据结构化证据生成模板化摘要。

## 快速开始

### 1. 创建环境

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
```

Linux/macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，设置本地数据库连接：

```env
CHROMA_MODE=http
CHROMA_HOST=127.0.0.1
CHROMA_PORT=8001
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-secure-local-password
```

如需构建经过验证的概念图谱或使用 LLM 生成与 NLI 校验回答，再设置：

```env
DEEPSEEK_API_KEY=your-deepseek-api-key
```

Neo4j 密码至少应为 8 个字符。

### 2. 启动 ChromaDB 和 Neo4j

```bash
docker compose up -d --wait chromadb neo4j
```

服务只监听本机：

- ChromaDB：`127.0.0.1:8001`
- Neo4j Browser：`http://127.0.0.1:7474`
- Neo4j Bolt：`127.0.0.1:7687`

数据分别保存在 Docker 命名卷 `chroma_data` 和 `neo4j_data` 中。

检查容器状态：

```bash
docker compose ps
```

### 3. 构建数据

```bash
python run_pipeline.py --stage all
```

`all` 会依次执行：

1. `parse`：解析培养方案和教学大纲，抽取并严格对齐显式先修课程。
2. `concept`：从课程目标和教学内容中提取概念，执行别名规范化、候选依赖
   评分和多次验证，并把规范概念回填到课程片段。
3. `graph`：生成课程图和概念依赖图摘要，并写入本地 Neo4j Community。
4. `embed`：使用包含规范概念的增强文本生成向量，并写入 ChromaDB 的
   `zstt_chunks` collection；检索结果仍返回原始课程文本。

无法唯一对齐的先修名称不会写成硬依赖边，而会保存到
`outputs/unresolved_prerequisites.json`，供人工确认。

概念阶段生成以下可审计产物：

- `concept_registry.json`：规范概念、别名、学科、Bloom 层级和来源课程。
- `concept_candidate_edges.json`：融合语义、课程顺序和规则信号的候选边。
- `concept_verified_edges.json`：经过多次关系判断的验证结果。
- `concept_aliases.json`：规范概念与原始别名映射。
- `concept_extraction_cache.json`：按文本、模型和提示版本缓存的抽取结果。
- `concept_validation_cache.json`：按候选边、模型、投票配置和提示版本缓存的
  完整 LLM 验证结果；供应商异常产生的降级结果不会进入缓存。

`graph` 阶段只接受 canonical registry 与 verified edge 两份权威产物；任一缺失
或本轮 `concept` 失败时会跳过 Neo4j 更新，不会用旧版 `concepts.json` 或空列表
覆盖已有概念图。

概念 ID 由规范名称稳定生成。只有验证来源为完整 LLM 投票或人工审核、布尔值
严格为 `requires=true`、置信度达到
`CONCEPT_VERIFIED_MIN_CONFIDENCE`、端点存在且不会形成环的受支持关系才会写入
Neo4j。概念阶段需要 `DEEPSEEK_API_KEY`；未配置时管线会跳过该阶段并保留已有
概念产物，避免用规则降级结果覆盖已验证图谱。候选验证数量由
`CONCEPT_MAX_VERIFICATION_CANDIDATES` 设置上限；Wikipedia 规范名查询默认关闭，
只有显式设置 `CONCEPT_WIKIPEDIA_ENABLED=true` 才会启用。正式概念阶段也会在
配置的本地 Embedding 模型不可用时停止并保留旧快照；需要确定性 Hash 时应显式
设置 `EMBEDDING_PROVIDER=hash`，产物清单会据此记录实际后端。
正式快照还要求至少达到 `CONCEPT_MIN_EXTRACTION_COVERAGE` 的片段抽取覆盖率；
模型返回 schema 无效数组或整轮覆盖率过低时不会发布空图。

Neo4j 写入使用 `ZSTT_Course`、`ZSTT_Concept`、`ZSTT_Chunk` 专属标签作为
匹配和唯一约束命名空间，默认只替换带有 `managed_by="zsttSystem"` 的节点，
不会接管其他应用仅使用通用 `Course`、`Concept`、`Chunk` 标签的同 ID 节点。
旧版本首次迁移且数据库为本项目专用库时，可临时设置
`RESET_CONCEPT_SUBGRAPH=true` 清理未标记的旧 `Course`、`Concept`、`Chunk`
节点；该选项会删除这些标签下的全部节点，不应在共享数据库中启用。

单独运行阶段：

```bash
python run_pipeline.py --stage parse
python run_pipeline.py --stage concept
python run_pipeline.py --stage graph
python run_pipeline.py --stage embed
```

增量解析：

```bash
python run_pipeline.py --stage parse --incremental
```

强制全量解析：

```bash
python run_pipeline.py --stage parse --force
```

### 4. 启动 API

Windows：

```powershell
.\start_all.bat
```

Linux/macOS：

```bash
./run_all.sh              # 容器跑 Chroma + Neo4j，API 用本地 .venv
./run_all.sh --docker     # 含 API 在内全部容器化
./run_all.sh --pipeline   # 启动前先跑一遍离线流水线
```

两个脚本都从脚本所在目录启动 ChromaDB、Neo4j 和 FastAPI，不依赖本机
绝对路径，并等待健康检查通过后再启动 API。

全容器方式只需 Docker，不需要本地 Python 环境：

```bash
docker compose --profile pipeline run --rm pipeline   # 离线流水线
docker compose up -d --wait chromadb neo4j api        # 在线服务
```

镜像里的应用以非 root 用户运行，嵌入模型缓存挂在 `model_cache` 卷上，
不打进镜像。

### 5. 验证

- 演示页：`http://127.0.0.1:8000/`
- OpenAPI：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

查询示例：

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"管理运筹学主要学什么？\"}"
```

## 不使用 Docker 的 ChromaDB

将 `.env` 改为：

```env
CHROMA_MODE=local
VECTOR_DB_PATH=chroma_data
```

随后直接运行管线和 API。ChromaDB 会持久化到项目目录下的
`chroma_data/`，该目录已被 Git 忽略。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 本地演示页 |
| GET | `/health` | ChromaDB、向量集合和 Neo4j 状态 |
| POST | `/query` | 自动路由问答 |
| GET | `/courses/{course_code}` | 课程信息 |
| GET | `/courses/{course_code}/graph` | 课程专属子图 |
| GET | `/courses/{course_code}/dependencies?depth=2&max_nodes=30&program_name=...` | 先修子图与选课路径 |
| GET | `/dependency?query=...` | 课程依赖查询 |
| POST | `/feedback` | 记录用户反馈 |

`POST /query` 可显式传入 `persona`，支持 `student`、`teacher` 和
`visitor`，默认值为 `student`。角色只调整检索证据优先级、回答深度和组织
方式，不改变课程事实。例如：

```json
{
  "query": "管理运筹学主要学什么？",
  "persona": "student"
}
```

课程先修子图只接受 `depth=1/2/3`，最多返回 30 个节点。边的统一方向是
`先修课程 --PREREQUISITE_OF--> 后续课程`；发生裁剪时响应中的
`truncated` 为 `true`，并通过 `total_nodes` 返回裁剪前节点数。

依赖响应中的 `plan` 使用完整硬先修祖先图进行 DAG 检查和拓扑分层，不受
展示子图的二跳裁剪影响。`stage` 表示建议学习层级，不等同于学校真实学期；
`official_semester` 才是培养方案中的开课学期。同一课程属于多个培养方案时，
`plan.status` 会返回 `program_required`，需要通过 `program_name` 明确选择，
系统不会混用不同方案的学期。

演示页使用项目内固定的 Mermaid 11.16.0 渲染课程依赖图，配置
`securityLevel: "strict"`，不依赖公共 CDN。图形失败时仍保留课程边列表。

## 查询路由

- `fact`：优先匹配培养方案中的课程结构化信息，可在一个问题中同时回答
  学分、学时等多个字段。
- `content`：先识别课程名称或代码，再按课程过滤 ChromaDB，合并课程大纲
  和培养方案证据。
- `dependency`：明确询问课程先修关系时，优先读取课程大纲中的先修字段；
  其他知识依赖问题使用本地 Neo4j。
- `catalog`：根据专业、主修或辅修类型和课程类别查询培养方案。
- `hybrid`：组合 ChromaDB 结构化证据、Neo4j 路径和 LLM 生成。

意图分类返回**多标签与置信度**（`QueryRouter.classify_intent()`）：优先级链
决定主意图，`confidence` 是主意图命中的模式数占全部命中数的比例。
命中两个可合并意图（`fact+dependency`、`content+dependency`、
`catalog+dependency`）时分别调用对应 handler，按小标题拼接、引用去重合并，
`query_type` 记为 `dependency+fact`；命中多个不可合并意图且置信度低于 0.34 时
转 `hybrid` 兜底。路由标签、置信度与各意图得分写入 `query_log.jsonl`，
可用 `eval/run_eval.py` 生成混淆矩阵。

内容与混合查询使用相关度阈值和轻量重排。未识别到明确课程时会提高相关度
要求，避免把弱相关课程强行作为答案；`fact` 路径退回向量检索时，若 Top-1
证据与问题没有任何区分性词面重合，会直接拒答而不是返回原文片段。

重排为 `向量分 × RERANK_WEIGHT_VECTOR + 词面分 × RERANK_WEIGHT_LEXICAL
+ 章节加权 + 来源加权`。词面分默认使用 BM25（IDF 由 embed 阶段离线统计到
`outputs/lexical_stats.json`），统计文件缺失时回退为 bigram 重合率。
权重全部可配置，`eval/tune_rerank.py` 负责网格搜索与敏感性曲线。

生成器接收结构化证据项，回答顺序为：

1. 直接回答问题。
2. 列出核心内容。
3. 列出资料来源。

LLM 暂时不可用或返回空内容时，系统会生成模板化摘要，不会返回原始
evidence、JSON 或 HTTP 500。没有足够可靠证据时会明确说明无法回答。

配置 LLM 后，内容和混合回答会进入逐事实句验证流程。系统用检索证据重新生成
来源列表，并排除这些确定性来源条目，只检查回答中的事实性陈述；未通过时按
`NLI_MAX_RETRIES` 定向重写，
重试后仅保留已支持的句子，存在矛盾或完全缺乏支持时安全拒答。验证详情通过
`metadata.nli_status`、`metadata.nli_details` 和 `metadata.nli_attempts` 记录，
并进入查询日志供后续人工审核。当前实现使用配置的 `JUDGE_MODEL` 作为
LLM-as-judge groundedness 校验器。在线返回策略比比例阈值更严格：所有最终保留
的事实句都必须被判断为 `Entailment`；`NLI_ENTAILMENT_THRESHOLD` 仍用于底层校验
函数和离线评测。

生成、NLI 上下文、答案内来源和 API `citations` 共用同一组最多 10 条证据，
避免返回模型与校验器没有见过的引用。

`citations` 与答案分离，仅向客户端公开：

- `course_name`
- `course_code`
- `section`
- `source_file`

查询 ID、检索分数、`chunk_id` 和内部元数据不会作为引用内容展示。

## 维护反馈数据

生成待审核样本：

```bash
python maintenance/active_learning_sampler.py
```

应用人工修正：

```bash
python maintenance/retraining_updater.py --evaluate
```

修正课程片段后，维护脚本会更新本地 JSON，并把修正后的语料**建成新的
collection 版本** `zstt_chunks_v<n>`，同时重建 BM25 统计。在线服务通过
`outputs/collection_alias.json` 指针解析实际读取的 collection：

- `--evaluate`：在金标集上分别评测旧版本与新版本并打印 before/after 对比，
  指标下降时不切换别名；
- `--force`：指标下降也强制切换；
- `--rollback`：把别名指回上一个版本（一次指针写入，不需要重跑解析）。

## 评测

量化评测集与指标体系位于 `eval/`，详见 [`eval/README.md`](eval/README.md)。

```bash
python eval/build_seed_dataset.py          # 生成 170 题种子集
python eval/run_eval.py --stages routing   # 不需要索引即可评路由
python eval/run_eval.py                    # 索引就绪后跑全量三阶段
python eval/ablation.py --with-answers     # 检索消融
python eval/tune_rerank.py                 # 重排权重网格搜索
python eval/persona_overlap.py             # 角色检索差异
python eval/concept_eval.py --make-template  # 概念抽取金标模板
```

四组指标：路由准确率与混淆矩阵、检索 Recall@k / MRR、答案要点命中率与引用
正确率、拒答与误拒率。报告写入 `eval/reports/`（JSON 明细 + Markdown 表格）。
三个阶段各自独立降级：没有向量索引仍可评路由，没有 `DEEPSEEK_API_KEY` 会评测
模板降级路径并在报告头部标注后端可用性。

## 开发与测试

```bash
python -m pip install -r requirements-dev.txt
ruff check src tests eval run_pipeline.py
python -m compileall -q src maintenance tests eval run_pipeline.py
pytest -q
```

完整测试包含概念清洗、稳定 ID、候选边 fail-closed、Neo4j 契约、NLI
重写/拒答及原有质量回归。其中 `tests/test_quality_regression.py` 固定覆盖：

- “管理运筹学主要学什么”。
- “信管专业核心课程有哪些”。
- “管理运筹学多少学分、多少学时”。
- “信息组织基础有哪些先修课程”。
- 无答案和弱相关问题。
- 页面不得显示字面量 `\n`、原始 JSON 或内部检索字段。

只运行质量回归集：

```bash
pytest tests/test_quality_regression.py -q
```

GitHub Actions 会在 Python 3.11 和 3.12 上安装开发依赖、检查依赖一致性、
编译 Python 源码、运行关键 Ruff 静态错误检查并执行完整测试。

## 常见问题

### API 报端口 8000 已被占用

Windows 错误 `10048` 或 `[Errno 10048]` 表示已经有进程监听
`127.0.0.1:8000`。新版 `start_all.bat` 会先检查 `/health`：

- 已有健康的 zsttSystem 实例时直接复用。
- 端口由异常实例或其他程序占用时停止启动，并给出提示。

检查占用进程：

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen
```

如果确认该 PID 是需要关闭的旧 zsttSystem 进程，可执行：

```powershell
Stop-Process -Id <PID>
```

然后重新运行 `start_all.bat`。

Hugging Face 的 unauthenticated request 提示只是模型下载限流警告，不会导致
端口绑定失败。需要更高下载限额时可配置 `HF_TOKEN`。

### ChromaDB 无法连接

```bash
docker compose ps
docker compose logs chromadb
```

确认 `.env` 中 `CHROMA_MODE=http`、`CHROMA_HOST=127.0.0.1`、
`CHROMA_PORT=8001`。

### ChromaDB 中没有数据

执行：

```bash
python run_pipeline.py --stage embed
```

然后访问 `/health`，确认 `chunk_count` 大于 0。

### Neo4j 不可用

```bash
docker compose ps neo4j
docker compose logs neo4j
```

确认 `.env` 中使用 `bolt://127.0.0.1:7687`，且 API 使用的密码与
Docker Compose 初始化密码一致。首次创建 `neo4j_data` 后，修改
`NEO4J_PASSWORD` 不会自动修改已有数据库密码；需要在 Neo4j 中修改密码，
或明确删除本地开发卷后重新初始化。

内容检索不依赖 Neo4j；数据库停止时，只有 dependency 查询和 hybrid 图
路径会降级。

### Embedding 模型无法下载

首次构建需要下载 `.env` 中配置的 SentenceTransformer 模型。可提前缓存
模型，或在测试环境设置：

```env
EMBEDDING_PROVIDER=hash
```

Hash 模式仅用于测试，不建议用于正式检索。

## 许可证

代码采用 [MIT License](LICENSE)。

`data/` 下教学材料的版权和分发授权不由 MIT 软件许可证自动覆盖；公开
发布或再分发这些材料前，应由数据提供方确认授权和隐私要求。
