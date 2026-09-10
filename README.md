<div align="center">

# zsttSystem

### 培养方案问答与选课路径规划 —— 答不上来的时候，它会说答不上来

[![Python 3.11 | 3.12](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-RAG--KG-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22A699.svg)](LICENSE)

[30 秒看懂](#它解决什么问题) · [快速开始](#60-秒跑起来) · [它答得准吗](#它答得准吗) ·
[换一所学校](#换一所学校怎么办) · [工作原理](#工作原理) · [API](#api)

</div>

---

## 它解决什么问题

在教务问答这件事上，**答错比答不出来严重得多**。

学生问「这门课多少学分」，得到一个编出来的数字，他真的会照着选错课。所以这个系统
的第一目标不是「什么都能答」，而是**没有可靠证据时闭嘴**。

评测里抓到过一个真实的例子。问一门根本不存在的课：

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"《火星种植学》有几学分？"}'
```

早期版本会退回向量检索，把 Top-1 片段原文当答案返回——于是它用**某门真实课程的
教材列表**回答了一门虚构的课。现在的回答是：

```json
{
  "answer": "根据当前知识库，未找到足够相关且可靠的资料来回答这个问题。",
  "citations": [],
  "query_type": "fact",
  "status": "succeeded"
}
```

拒答是正常返回，不是报错。**「我不知道」在这个场景里比「编一个」值钱。**

## 两件它做得跟别人不一样

### 一、拒答有三道闸，不是一句提示词

大多数 RAG 的「不知道就别答」是写在 prompt 里的一句请求。这里是三层硬约束：

1. **词面门槛**：`fact` 路径退回向量检索时，若 Top-1 证据与问题没有任何区分性词面
   重合（BM25 词面分为 0），直接拒答，而不是返回原文片段。
2. **逐句校验**：配置 LLM 后，内容与混合类回答会进入逐事实句 groundedness 验证
   （`JUDGE_MODEL` 作 LLM-as-judge）。系统用检索证据重新生成来源列表并排除这些
   确定性条目，只检查回答中的事实性陈述。
3. **在线策略比阈值更严**：所有最终保留的事实句**必须全部判定为 `Entailment`**
   才放行。未通过时按 `NLI_MAX_RETRIES` 定向重写；重试后仅保留已支持的句子；
   存在矛盾或完全缺乏支持时安全拒答。

全过程写入 `metadata.nli_status` / `nli_details` / `nli_attempts` 并进查询日志，
可人工复核。LLM 不可用时退化为模板化摘要，**不会返回原始 evidence、JSON 或 500**。

### 二、给决策，不只给答案

学生真正的问题往往不是「这课学什么」，而是**「我该先学什么」**。

```
GET /courses/{course_code}/dependencies?depth=2&program_name=信息管理与信息系统
```

返回的 `plan` 用**完整硬先修祖先图**做 DAG 检查和拓扑分层，不受展示子图二跳裁剪的
影响。三个刻意的设计：

- `stage` 是**建议学习层级**，`official_semester` 才是培养方案里的真实开课学期。
  两者分开，不让算法推荐冒充学校规定。
- 同一门课属于多个培养方案时返回 `plan.status = "program_required"`，要求用
  `program_name` 明确选择——**系统不会混用不同方案的学期**。
- 边方向统一为 `先修课程 --PREREQUISITE_OF--> 后续课程`；裁剪时 `truncated` 为
  `true`，并用 `total_nodes` 返回裁剪前的节点数。

这也是知识图谱这条腿存在的理由：**纯向量检索做不出拓扑排序。**

## 60 秒跑起来

需要 Python 3.11 / 3.12 和 Docker Desktop。

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cp .env.example .env                                  # Windows: copy .env.example .env

docker compose up -d --wait chromadb neo4j            # 只监听本机
python run_pipeline.py --stage all                    # 离线管线
./run_all.sh                                          # Windows: .\start_all.bat
```

- 演示页 `http://127.0.0.1:8000/`
- OpenAPI `http://127.0.0.1:8000/docs`
- 健康检查 `http://127.0.0.1:8000/health`

`.env` 里必填的只有本地数据库连接：

```env
CHROMA_MODE=http
CHROMA_HOST=127.0.0.1
CHROMA_PORT=8001
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-secure-local-password    # 至少 8 位
```

**`DEEPSEEK_API_KEY` 是可选的。** 不配也能跑：系统会根据结构化证据生成模板化摘要。
配了才有 LLM 生成、NLI 校验和概念图谱构建。

全容器方式不需要本地 Python 环境：

```bash
docker compose --profile pipeline run --rm pipeline   # 离线流水线
docker compose up -d --wait chromadb neo4j api        # 在线服务
```

镜像内以非 root 运行，嵌入模型缓存挂在 `model_cache` 卷上，不打进镜像。

## 它答得准吗

`eval/` 是一套四组指标的量化评测体系：路由准确率与混淆矩阵、检索 Recall@k / MRR、
答案要点命中率与引用正确率、拒答与误拒率。报告写入 `eval/reports/`（JSON 明细 +
Markdown 表格）。

**已经测到的：路由准确率 0.8118 → 0.9529**（170 题金标集，其中无答案题 25 题）。
路由分类不依赖向量索引和 LLM，所以这个数字是干净的。

| 类别 | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| fact | 1.00 | 0.8889 | 0.9412 | 54 |
| content | 1.00 | 0.9565 | 0.9778 | 46 |
| dependency | 0.9574 | 1.00 | 0.9783 | 45 |
| catalog | 1.00 | 1.00 | 1.00 | 25 |

提升来自评测跑起来后抓到的两个具体缺陷：`(必修|选修)` 属于 fact 模式且 fact 优先级
高于 catalog，导致 10 道目录题被误判（catalog 召回 0.60 → 1.00）；路由只认字面
「开课学期」，「第几学期开课」「什么学期上」全都漏掉（fact 精确率 0.77 → 1.00）。

**还没测到的，这里说清楚：** 本机 `chroma_data/` 为空、概念图谱尚未生成，所以检索
Recall@k、答案要点命中率、引用正确率的**真实数字目前不存在**。已有的端到端数值都是
在 hash 伪向量、无 LLM、无图谱的降级配置下跑出的**下限值**，只能证明评测能抓问题，
不能当作系统的实际表现。三个阶段各自独立降级——没有向量索引仍可评路由，没有
`DEEPSEEK_API_KEY` 会评测模板降级路径，并在报告头部标注后端可用性。

复现路由结论（秒级，不需要索引）：

```bash
python eval/build_seed_dataset.py
python eval/run_eval.py --stages routing --tag routing-check
```

## 换一所学校怎么办

代码里**没有任何学校名、专业名的硬编码**。专业名与方案类型全部从解析结果里自动
学习，只有口语缩写需要配置：

```dotenv
PROGRAM_ALIASES=信管=信息管理与信息系统
```

换语料有两种方式（详见 [`data/README.md`](data/README.md)）：直接拷进 `data/` 后跑
流水线，或者用导入接口。导入接口**默认关闭**，设置 `ADMIN_TOKEN` 后才启用：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/admin/data` | 当前语料清单 |
| POST | `/admin/data/import` | 上传 .docx / .xlsx（multipart，字段名 `files`） |
| POST | `/admin/data/reindex` | 后台重新解析并重建索引，立即返回 202 |
| GET | `/admin/data/reindex` | 查看重建进度 |

```bash
curl -X POST http://127.0.0.1:8000/admin/data/import \
  -H "Authorization: Bearer $ADMIN_TOKEN" -F "files=@IM104档案学概论.docx"
curl -X POST http://127.0.0.1:8000/admin/data/reindex \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

**重建成功后热替换进正在运行的服务，不需要重启。**

### 换错了能退回来

人工修正课程片段后，维护脚本不会原地重建当前集合，而是**建一个新版本**
`zstt_chunks_v<n>`，同时重建 BM25 统计。在线服务通过 `outputs/collection_alias.json`
指针解析实际读取的 collection：

```bash
python maintenance/active_learning_sampler.py       # 生成待审核样本
python maintenance/retraining_updater.py --evaluate # 金标集上对比新旧版本
```

- `--evaluate`：打印 before/after 对比，**指标下降就不切换别名**
- `--force`：指标下降也强制切换
- `--rollback`：把别名指回上一个版本（一次指针写入，不需要重跑解析）

## 工作原理

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

### 查询路由

意图分类返回**多标签与置信度**（`QueryRouter.classify_intent()`）：优先级链决定主
意图，`confidence` 是主意图命中的模式数占全部命中数的比例。

| 类型 | 做什么 |
| --- | --- |
| `fact` | 优先匹配培养方案中的结构化信息，可在一个问题里同时回答学分、学时等多个字段 |
| `content` | 先识别课程名称或代码，再按课程过滤 ChromaDB，合并大纲与培养方案证据 |
| `dependency` | 明确问先修关系时优先读大纲的先修字段；其他知识依赖问题走 Neo4j |
| `catalog` | 按专业、主修/辅修类型和课程类别查培养方案 |
| `hybrid` | 组合结构化证据、图路径与 LLM 生成 |

命中两个可合并意图（`fact+dependency`、`content+dependency`、`catalog+dependency`）
时分别调用对应 handler，按小标题拼接、引用去重合并，`query_type` 记为
`dependency+fact`；命中多个不可合并意图且置信度低于 0.34 时转 `hybrid` 兜底。

路由标签、置信度与各意图得分写入 `query_log.jsonl`，`eval/run_eval.py` 据此生成
混淆矩阵。

### 检索与重排

```text
向量分 × RERANK_WEIGHT_VECTOR + 词面分 × RERANK_WEIGHT_LEXICAL + 章节加权 + 来源加权
```

词面分默认使用 BM25，IDF 由 embed 阶段离线统计到 `outputs/lexical_stats.json`
（实测语料 2677 文档、39843 词，剪枝后 18889 条、0.25 MB、0.14 秒）；统计文件缺失时
自动回退为 bigram 重合率，新检出的仓库不会崩。

权重全部可配置，`eval/tune_rerank.py` 负责网格搜索与敏感性曲线——当前最优为
`vector 0.75 / lexical 0.15 / section_boost 0.08`，曲线直接回答「为什么是 0.75」。

语料实测有个有意思的现象：文档频率最高的词面正是「学时 / 名称 / 教学 / 内容 / 主要」
（出现在 60–73% 的 chunk 里），IDF 加权后它们的贡献被压到接近 0。

未识别到明确课程时会提高相关度要求，避免把弱相关课程强行当作答案。

### 生成与引用

生成器接收结构化证据项，回答顺序固定为：直接回答问题 → 列出核心内容 → 列出资料来源。

生成、NLI 上下文、答案内来源和 API `citations` **共用同一组最多 10 条证据**，避免
返回模型与校验器都没见过的引用。`citations` 与答案分离，仅公开四个字段：

`course_name` · `course_code` · `section` · `source_file`

查询 ID、检索分数、`chunk_id` 和内部元数据不会作为引用内容展示。

### 离线管线的 fail-closed

`concept` 阶段从课程目标和教学内容提取概念，执行别名规范化、候选依赖评分和多次
验证，并把规范概念回填到课程片段。它生成六份可审计产物，包括
`concept_registry.json`（规范概念、别名、学科、Bloom 层级、来源课程）、
`concept_candidate_edges.json` 和 `concept_verified_edges.json`。

写入 Neo4j 的门槛很高，**任何一步不确定都宁可不写**：

- `graph` 阶段只接受 canonical registry 与 verified edges 两份权威产物；任一缺失或
  本轮 `concept` 失败时跳过 Neo4j 更新，**不会用旧版或空列表覆盖已有概念图**。
- 只有验证来源为完整 LLM 投票或人工审核、布尔值严格为 `requires=true`、置信度达到
  `CONCEPT_VERIFIED_MIN_CONFIDENCE`、端点存在且不成环的关系才会写入。
- 概念阶段需要 `DEEPSEEK_API_KEY`；未配置时跳过该阶段并保留已有产物，避免用规则
  降级结果覆盖已验证图谱。
- 正式快照要求达到 `CONCEPT_MIN_EXTRACTION_COVERAGE` 的片段抽取覆盖率；schema 无效
  或覆盖率过低时不会发布空图。
- 无法唯一对齐的先修名称不会写成硬依赖边，而是留在
  `outputs/unresolved_prerequisites.json` 供人工确认。

Neo4j 写入使用 `ZSTT_Course` / `ZSTT_Concept` / `ZSTT_Chunk` 专属标签作为命名空间，
默认只替换带 `managed_by="zsttSystem"` 的节点，不会接管其他应用仅用通用 `Course`、
`Concept`、`Chunk` 标签的同 ID 节点。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 本地演示页 |
| GET | `/health` | ChromaDB、向量集合和 Neo4j 状态 |
| POST | `/query` | 自动路由问答 |
| GET | `/courses/{course_code}` | 课程信息 |
| GET | `/courses/{course_code}/graph` | 课程专属子图 |
| GET | `/courses/{course_code}/dependencies` | 先修子图与选课路径 |
| GET | `/dependency?query=...` | 课程依赖查询 |
| POST | `/feedback` | 记录用户反馈 |

`POST /query` 可显式传入 `persona`（`student` / `teacher` / `visitor`，默认
`student`）。角色只调整检索证据优先级、回答深度和组织方式，**不改变课程事实**。

```json
{ "query": "管理运筹学主要学什么？", "persona": "student" }
```

先修子图只接受 `depth=1/2/3`，最多返回 30 个节点。

query 长度限制为 1–500 字符（`API_MAX_QUERY_CHARS`）。`/query`、`/dependency`、
`/admin`、`/feedback` 四条路径受每客户端每分钟 60 次的滑动窗口限流保护
（`API_RATE_LIMIT_PER_MINUTE`，设为 0 即关闭；超限返回 429 + `Retry-After`）。
CORS 白名单由 `API_CORS_ORIGINS` 控制，留空则完全不启用。

演示页的依赖图用的是仓库内自带的 Mermaid（`src/static/vendor/mermaid.min.js`），
配置 `securityLevel: "strict"`，**不依赖公共 CDN**；渲染失败时仍保留课程边列表。

## 附录

### 不使用 Docker 的 ChromaDB

```env
CHROMA_MODE=local
VECTOR_DB_PATH=chroma_data
```

随后直接运行管线和 API，ChromaDB 会持久化到项目目录下的 `chroma_data/`（已被 Git 忽略）。

### 管线单阶段运行

```bash
python run_pipeline.py --stage parse       # 解析并严格对齐显式先修课程
python run_pipeline.py --stage concept     # 概念抽取、别名规范化、依赖验证
python run_pipeline.py --stage graph       # 写入 Neo4j
python run_pipeline.py --stage embed       # 生成向量写入 zstt_chunks

python run_pipeline.py --stage parse --incremental   # 增量解析
python run_pipeline.py --stage parse --force         # 强制全量
```

`embed` 使用包含规范概念的增强文本生成向量，但**检索结果仍返回原始课程文本**。

### 完整评测

```bash
python eval/run_eval.py                      # 索引就绪后跑全量三阶段
python eval/ablation.py --with-answers       # 检索消融
python eval/tune_rerank.py                   # 重排权重网格搜索
python eval/persona_overlap.py               # 角色检索差异
python eval/concept_eval.py --make-template  # 概念抽取金标模板
```

详见 [`eval/README.md`](eval/README.md)。

### 开发与测试

```bash
python -m pip install -r requirements-dev.txt
ruff check src tests eval run_pipeline.py
pytest -q
```

`tests/test_quality_regression.py` 固定覆盖六项：「管理运筹学主要学什么」「信管专业
核心课程有哪些」「管理运筹学多少学分、多少学时」「信息组织基础有哪些先修课程」、
无答案与弱相关问题，以及**页面不得显示字面量 `\n`、原始 JSON 或内部检索字段**。

```bash
pytest tests/test_quality_regression.py -q
```

GitHub Actions 在 Python 3.11 和 3.12 上安装开发依赖、检查依赖一致性、编译源码、
运行关键 Ruff 静态错误检查并执行完整测试。

### 常见问题

<details>
<summary>API 报端口 8000 已被占用</summary>

Windows 错误 `10048` 表示已有进程监听 `127.0.0.1:8000`。`start_all.bat` 会先检查
`/health`：已有健康实例时直接复用；端口被异常实例或其他程序占用时停止启动并提示。

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen
Stop-Process -Id <PID>
```

Hugging Face 的 unauthenticated request 提示只是模型下载限流警告，不会导致端口绑定
失败。需要更高下载限额时配置 `HF_TOKEN`。

</details>

<details>
<summary>ChromaDB 无法连接 / 没有数据</summary>

```bash
docker compose ps
docker compose logs chromadb
```

确认 `.env` 中 `CHROMA_MODE=http`、`CHROMA_HOST=127.0.0.1`、`CHROMA_PORT=8001`。
没有数据时执行 `python run_pipeline.py --stage embed`，然后访问 `/health` 确认
`chunk_count` 大于 0。

</details>

<details>
<summary>Neo4j 不可用</summary>

```bash
docker compose ps neo4j
docker compose logs neo4j
```

确认使用 `bolt://127.0.0.1:7687`，且 API 密码与 Docker Compose 初始化密码一致。首次
创建 `neo4j_data` 后修改 `NEO4J_PASSWORD` 不会自动改已有数据库密码。

**内容检索不依赖 Neo4j**；数据库停止时只有 dependency 查询和 hybrid 图路径会降级。

</details>

<details>
<summary>Embedding 模型无法下载</summary>

首次构建需要下载 `.env` 中配置的 SentenceTransformer 模型。可提前缓存，或在测试环境
设置 `EMBEDDING_PROVIDER=hash`。**Hash 模式仅用于测试，不可用于正式检索。**

</details>

### 技术栈

- **Web：** FastAPI（查询、课程、依赖、反馈与管理接口）
- **向量：** ChromaDB，`zstt_chunks` collection，SentenceTransformer 本地嵌入
- **图：** Neo4j Community，`ZSTT_*` 标签命名空间
- **词面：** 自建 BM25（Robertson IDF，离线统计），rapidfuzz 无关
- **生成与校验：** DeepSeek 或任意 OpenAI Chat Completions 兼容模型
- 不依赖 LightRAG，也不需要单独的 Embedding HTTP 服务

## 许可证

代码采用 [MIT License](LICENSE)。

`data/` 下教学材料的版权和分发授权**不由 MIT 软件许可证自动覆盖**；公开发布或再分发
这些材料前，应由数据提供方确认授权和隐私要求。
