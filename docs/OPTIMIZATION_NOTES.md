# 优化说明：对照《论文与优化建议》的落地记录

对照文档：`D:\Projects\zsttsystem\论文与优化建议.md`（2026-09-05）
实施对象：`D:\python\zsttSystem1.1\zsttSystem`
基线：优化前 99 项 pytest 全通过；优化后 141 项全通过。

本文件只记录**代码侧**的改动。文档第一、二部分（拟题、文献）不涉及代码。

---

## 一、总览

| 建议 | 状态 | 落点 |
| --- | --- | --- |
| P0-1 概念抽取重做 | **部分已存在 + 已改造**（见下方说明） | `src/data_processing/concept_extractor.py`、`eval/concept_eval.py` |
| P0-2 量化评测集与指标 | ✅ 已建成并跑通 | `eval/` 全套 |
| P0-3 反馈闭环有效性证据 | ✅ 版本化 + 别名切换 + before/after | `src/data_processing/collection_registry.py`、`maintenance/retraining_updater.py` |
| P1-1 重排魔数 | ✅ 权重进配置 + BM25 + 循环外提 | `src/utils/lexical.py`、`src/data_processing/lexical_stats.py`、`chroma_retriever.py`、`eval/tune_rerank.py` |
| P1-2 意图分类顺序敏感 | ✅ 多标签 + 置信度 + 多意图合并 | `src/online_service/query_router.py` |
| P1-3 persona 叙事对齐 | ✅ 提供测量脚本（结论待跑真实索引） | `eval/persona_overlap.py` |
| P2-1 Neo4j Chunk 冗余 | ✅ 改为可配置，默认关闭 | `graph_builder.py`、`run_pipeline.py` |
| P2-2 可复现性（仅 Windows） | ✅ Dockerfile + compose 应用服务 + `run_all.sh` | 根目录 |
| P2-3 数据授权 | ⚠️ 未动数据（见"未做的事"） | — |
| P2-4 API 加固 | ✅ 长度上限 + 限流 + CORS | `src/main.py`、`src/config.py` |

---

## 二、需要更正建议文档的一点：P0-1 的前提已变

建议文档基于 GitHub 上的版本，判断"概念抽取只有 54 行正则、`llm_client` 参数
从未使用"。**在本地 1.1 版本中，这条已经不成立**：

- 线上概念链路是 `ConceptNormalizer.extract_core_concepts()`
  （`concept_normalizer.py`，1489 行）：受限 prompt、JSON schema 约束、
  Bloom 层级、多轮投票、噪声名过滤、抽取/校验双缓存；
- `run_pipeline.py` 的 concept 阶段只调用它，**`concept_extractor.py` 已无任何
  import**，属于死代码。

因此没有按"把正则换成 LLM"去改（那会重复造轮子），而是：

1. **把 `concept_extractor.py` 明确定位成消融实验的规则基线**，并让它成为一个
   诚实的基线而不是稻草人：
   - 候选切分不再是"任意 2–16 字连续串"，而是先按标点、空白、连接词
     （与/和/及/以及/等/的）与教学动词（掌握/了解/要求…）切开；
   - 停用词表从 7 个扩到 40+，并加入章节结构过滤（第X章/第X节/复习/绪论…）；
   - 增加**跨课程文档频率过滤**：出现在超过 50% 课程里的短语判为教学套话；
   - 保留下来的候选按 TF-IDF 排序取前 N，结果可复现。
2. **修掉建议文档指出的 O(N²) 写盘 bug**：缓存文件原来在 for 循环内每门课全量
   重写一次，现在整批结束后只写一次，且无变更时完全不写。
3. `llm_client` 这个"签名里有、函数体不用"的参数直接删除——保留它才是误导。
4. 新增 `eval/concept_eval.py`：规则基线 vs LLM 抽取，同一金标集上给
   strict/lenient 两种匹配下的 micro/macro P/R/F1，直接产出论文表格。
   `--make-template` 会把两个抽取器的候选合并成待标注模板，标注人删噪声即可，
   比对着大纲从零录入快一个量级。

改造前后同一语料（101 门课、2677 个 syllabus chunk）的规则基线输出对比：

```
改造前：['档案管理基本理论', '第十章', '第一节', '勇往直前', '继续努力', ...]
改造后：['档案学', '档案管理基本理论', '档案工作', '国际档案治理', '档案事业', ...]
```

> ⚠️ 注意：`outputs/concepts.json` 是**旧 pipeline 留下的产物**，里面仍有
> "中山大学""本科课程教学大纲""课程名称"这类噪声。`concept_registry.json`
> 尚未生成（需要 `DEEPSEEK_API_KEY` 跑一次 `run_pipeline.py concept`）。
> `concept_eval.py` 在退回读 `concepts.json` 时会打印警告——**不要把这份旧产物
> 当成 LLM 抽取结果写进论文**。

---

## 三、P0-2 评测体系（本次最大增量）

新增 `eval/` 包：

```
eval/
├── README.md              标注规范 + 指标定义 + 实验清单
├── schema.py              GoldItem 定义与校验（非法金标直接报错，不静默打分）
├── metrics.py             路由/检索/生成/拒答四组指标 + Markdown 渲染
├── build_seed_dataset.py  从 courses.json 自动生成 170 题种子集
├── run_eval.py            三阶段主评测（routing / retrieval / answer）
├── ablation.py            检索消融（6 组）
├── tune_rerank.py         权重网格搜索 + 敏感性曲线
├── persona_overlap.py     角色检索重合率
├── concept_eval.py        概念抽取 P/R/F1
└── datasets/              gold_seed.json（已生成）、gold_questions.json（人工）
```

**种子集 170 题**：fact 40 / content 40 / dependency 40 / catalog 25 /
无答案 25（无答案题里 16 题是不存在的课程，9 题是真实课程但语料不记录的字段，
如上课教室、期末考试日期）。答案要点直接由培养方案字段推导，因此 fact /
dependency / catalog 三类**开箱即可自动判分**；content 类的答案要点和全部
chunk 级标注仍需人工补充（`eval/README.md` 写了分工建议）。

三个阶段各自独立降级：没有索引也能跑路由，没有 API Key 也能跑降级路径并在
报告头部标 `llm_available: false`——避免"换了环境却在比指标"。

### 评测一跑起来就抓到的问题（已修）

**1. 路由准确率 0.8118 → 0.9529**（同一份 170 题种子集）

| 混淆项 | 数量 | 原因 | 处理 |
| --- | --- | --- | --- |
| catalog→fact | 10 | `(必修\|选修)` 属于 fact 模式，且 fact 优先级高于 catalog | 把 CATALOG 提到 FACT 之前：带专业范围词的问题归目录，课程级问题不受影响 |
| fact→hybrid | 19 | 只认字面"开课学期"，"第几学期开课""什么学期上"全都漏 | 增加自然问法模式 |

catalog 召回率 0.60 → 1.00，fact 精确率 0.77 → 1.00。

**2. 拒答机制在 fact 路径上是失效的**

`_handle_fact` 在课程链接失败时会退回向量检索并**直接把 Top-1 chunk 原文当答案
返回**。评测里"《火星种植学》有几学分？"因此被回答成了某门课的教材列表。
现在要求 Top-1 命中至少有一个区分性词面重合（BM25 词面分 > 0），否则明确拒答。

**3. 培养方案目录题在辅修/大类方案上全部拒答**

"核心课程"只在部分方案里是显式类别；辅修微专业方案把所有课程记为"辅修课程"。
现在类别过滤按 核心 → 专业必修 → 不限 逐级放宽，已标注核心课的方案行为不变
（`test_catalog_query_uses_training_plan_memberships` 仍然通过）。

同一份种子集、同一套降级环境（hash 向量、无 LLM、无 Neo4j）下：

| 指标 | 修复前 | 修复后 |
| --- | --- | --- |
| 答案要点命中率 | 0.6476 | 0.7143 |
| 有答案但无引用的比例 | 0.1448 | 0.0000 |
| 正确拒答率 | 0.16 | 0.64 |
| 幻觉率（无答案题被作答） | 0.84 | 0.36 |

> 这组数字是**弱配置下限**（hash 伪向量、无 LLM、无图谱），不能直接写进论文当
> 主结果；它的价值是证明评测能抓问题、并给消融的下界。真实结论要在
> `EMBEDDING_PROVIDER=local` 且概念/图谱阶段跑完之后重跑。
> 剩余 0.36 的幻觉全部来自 content 路径用别的课程内容回答不存在的课程，
> 与 hash 向量强相关，**不建议在 hash 索引上调阈值**。

---

## 四、P1-1 检索重排

- 新增 `src/utils/lexical.py`：统一分词（拉丁词 + 中文 bigram）、Robertson IDF、
  Okapi BM25，并把 BM25 归一到 [0,1) 以便与向量分线性融合。
- 新增 `src/data_processing/lexical_stats.py`：在 embed 阶段离线统计文档频率写入
  `outputs/lexical_stats.json`（2677 文档、39843 词、剪枝后 18889 条、0.25 MB、
  0.14 秒）。统计缺失时自动退回旧的 bigram 重合率，新检出的仓库不会崩。
- `chroma_retriever.py`：
  - 权重全部来自 `config`（`RERANK_WEIGHT_*`、`RERANK_SECTION_BOOST`、
    `RERANK_LEXICAL_SCHEME`、`RERANK_BM25_K1/B`），并可通过 `RerankWeights`
    注入，消融与网格搜索因此不需要改环境变量；
  - `_query_terms(query)` 从 `for hit` 循环里提出来，每次查询算一次；
  - 命中结果新增 `lexical_score` 字段，便于错误分析。

语料实测：文档频率最高的词面正是"学时/名称/教学/内容/主要"（出现在 60–73%
的 chunk 里），IDF 加权后它们的贡献被压到接近 0——这正是建议文档预判的问题。

`eval/tune_rerank.py` 输出最优权重与三条敏感性曲线，直接回答"为什么是 0.75"。

---

## 五、P1-2 路由

- `QueryType` 里 `SIMPLE`/`COMPLEX` 两个**值重复的枚举别名**已删除（Python 会
  把它们变成别名而非独立成员，全仓库无引用）。
- `classify_intent()` 返回 `IntentPrediction{primary, labels, confidence, scores}`：
  - `primary` 仍按优先级链选出，**单意图行为与改造前完全一致**；
  - `confidence` = 主意图命中的模式数占全部命中数的比例（"证据份额"）；
  - `classify()` 保留为兼容入口。
- 多意图合并：`fact+dependency`、`content+dependency`、`catalog+dependency`
  三对会分别调用各自 handler，按【小标题】拼接、引用去重合并，
  `query_type` 记为 `dependency+fact`。建议文档里的例句
  "管理运筹学几学分，有哪些先修课？"现在两部分都答。
- 低置信兜底：多标签且不可合并、且置信度 < 0.34 时转 hybrid，
  元数据打 `low_confidence_fallback`。
- 路由元数据（labels/confidence/scores）写进 `query_log.jsonl`，混淆矩阵可以按
  置信度分层切片。

---

## 六、P0-3 反馈闭环

- 新增 `src/data_processing/collection_registry.py`：`zstt_chunks_v<n>` 版本命名 +
  `outputs/collection_alias.json` 别名指针（含切换历史与切换时的指标快照）。
  `ChromaRetriever` 启动时按别名解析实际集合，指针失效时回退到基础名。
- `retraining_updater.py`：
  - 人工修正后**建新版本**，不再原地重建当前集合；
  - `--evaluate` 在金标集上分别评测旧版本与新版本并打印 before/after 表，
    指标下降就**不切换**别名（`--force` 可强切）；
  - `--rollback` 按历史回退一步；
  - 同时重建 BM25 统计，避免词面分与新文本不同步。

这样"人工在环持续优化"才有可测量的证据，而不只是一条通的管道。

---

## 七、P1-3 persona

没有改实现，也没有改论文措辞——先给测量工具：`eval/persona_overlap.py` 在
真实持久化索引上计算三种 persona 两两的 Top-K 重合率、Jaccard 与 Top-1 变化率，
并在报告里写明判据（>0.9 应表述为"证据优先级调度"，<0.7 或 Top-1 变化率 >0.3
才支撑更强的角色感知表述）。**结论需要在真实向量索引上跑一次再定**。

---

## 八、P2 系列

- **P2-1**：`build_graph_records(..., include_chunk_nodes=)`，pipeline 默认
  `GRAPH_INCLUDE_CHUNK_NODES=false`。关闭时不再写入 Chunk 节点与
  CONTAINS/MENTIONS 边，也不再把整份语料文本复制进 Neo4j（原来
  `write_neo4j` 会把 chunk 全文写入图库）。打开时 Chunk 节点会带上
  course_code/section_type/source_type 等可供图查询的属性，而不是只有 id。
- **P2-2**：新增 `Dockerfile`（非 root、健康检查、模型缓存走挂载卷）、
  compose 增加 `api` 服务与 `pipeline` 一次性任务、新增 `run_all.sh`
  （`--docker` 全容器、`--pipeline` 先跑离线流水线）。
  ChromaDB 也补了 healthcheck，`depends_on` 才真正等到就绪。
- **P2-4**：`/query` 的 query 字段 `min_length=1, max_length=500`（可配置）、
  `/dependency` 同样限制、`/query` 与 `/dependency` 加每 IP 每分钟 60 次的
  滑动窗口限流（返回 429 + Retry-After）、CORS 白名单由 `API_CORS_ORIGINS` 控制
  （留空则完全不启用）。

---

## 九、未做的事

1. **P2-3 数据授权**：没有移动或删除 `data/` 下的任何原始 docx/xlsx。这属于
   授权与合规决策，应由团队决定保留哪 10 门课作为复现样例、原始件是否转私有仓
   库。代码侧不做既成事实。
2. **人工标注**：content 类答案要点、chunk 级金标、20 门课的概念金标——工具和
   模板都已就绪，标注本身需要人做。
3. **真实索引下的完整实验**：本机 `chroma_data/` 是空的，`concept_registry.json`
   也未生成。所有端到端数字都是在临时 hash 索引上跑的下限值。
4. **persona 的最终定调**：等 `persona_overlap.py` 在真实索引上的结果。

---

## 十、复现本次结论

```bash
# 1. 路由（不需要索引，秒级）
python eval/build_seed_dataset.py
python eval/run_eval.py --stages routing --tag routing-check

# 2. 建索引后的完整评测
python run_pipeline.py all          # 需要 DEEPSEEK_API_KEY 才有概念/图谱阶段
python eval/run_eval.py --tag full

# 3. 消融与权重
python eval/ablation.py --with-answers
python eval/tune_rerank.py --metric recall@5

# 4. persona 与概念
python eval/persona_overlap.py --top-k 5
python eval/concept_eval.py --make-template --courses 20   # 标注后再跑 concept_eval.py
```
