# 评测体系（eval/）

论文里能写进实验章节的所有数字都从这里产出：路由准确率与混淆矩阵、检索
Recall@k / MRR、答案与引用正确率、拒答与误拒率、检索消融、重排权重敏感性、
角色（persona）检索差异、概念抽取 P/R/F1。

没有这一层，任何"改好了"的说法都无法证明——所以先建评测集，再改模型。

---

## 0. 三分钟跑通

```bash
python eval/build_seed_dataset.py      # 生成 170 题种子集（不需要索引/API Key）
python eval/run_eval.py --stages routing   # 只评路由，秒级出结果
python eval/run_eval.py                # 索引就绪后跑全量三阶段
```

报告写到 `eval/reports/<时间戳>-<tag>.json` 与同名 `.md`。`.md` 可以直接贴进
论文初稿，`.json` 保留逐题明细，用于错误分析。

在 Docker Compose 容器里，`EVAL_REPORT_DIR` 已设为 `outputs/eval-reports`（已挂载），
报告不会随重建容器丢失，`/eval/latest` 也从这里读，不用再加 `--output-dir`：

```bash
docker compose exec api python eval/run_eval.py
```

`./eval/datasets` 会挂载到容器的 `/app/eval/datasets`，因此人工标注的评测集
也不会随镜像重建而消失。

---

## 1. 数据集

| 文件 | 来源 | 说明 |
| --- | --- | --- |
| `datasets/gold_seed.json` | `build_seed_dataset.py` 自动生成 | 种子集，可直接跑；**不入库**，见下 |
| `datasets/gold_questions.json` | 人工标注 | 存在时 `run_eval.py` 优先使用 |
| `datasets/concept_gold.json` | 人工标注 | 概念抽取金标（`concept_eval.py`） |

> **评测集不进 Git**：题目与答案要点由培养方案派生，含任课教师姓名等个人信息，
> 而本仓库是公开的。`eval/datasets/*.json` 已在 `.gitignore` 中。种子集用固定
> 随机种子生成，`python eval/build_seed_dataset.py` 可逐字节复现；人工标注的
> `gold_questions.json` / `concept_gold.json` 请在团队内部私下同步，等
> P2-3 数据授权问题定了再决定是否取消忽略。

### 单条题目的字段

```json
{
  "id": "fact-001",
  "question": "《信息管理学基础》有几学分？",
  "expected_route": "fact",
  "answerable": true,
  "answer_keys": ["学分为 3|3学分"],
  "gold_chunk_ids": [],
  "gold_course_codes": ["IM121"],
  "gold_section_types": [],
  "persona": "student",
  "source": "auto-seed",
  "notes": "field=credits"
}
```

> 示例特意选了学分字段。任课教师题同样在种子集里（`field=instructor`），
> 但姓名不写进入库的文档。

- `expected_route`：`fact` / `content` / `dependency` / `catalog` / `hybrid` / `structure` / `existence`
- `answer_keys`：**全部命中才算答对**；单个 key 内用 `|` 表示"任一等价说法"
- `gold_chunk_ids`：人工标注的证据 chunk；**有它就按 chunk 级判定**
- `gold_course_codes`：没有 chunk 标注时的课程级近似判定（报告会分别统计
  `chunk_level_items` / `course_level_items`，写论文时必须说明比例）
- `answerable: false`：无答案题，用于测拒答；这类题**不允许**有 `answer_keys`
- `focus_keys`：答案**第一句**应包含的片段（`|` 分隔候选），只给 `existence` 题用，
  拒答题也可以带；`expected_semester`：`semester-phrasing` 题的正确学期（1-8）。
  两者只服务于「答案切题率」，不影响 `answer_keys` 的判定

### 种子集自动生成的边界

自动生成能覆盖的：培养方案里已有确定值的字段题（学分/学时/学期/教师）、
先修关系题、培养方案目录题、无答案题。

**必须人工补的**：

1. `content` 类题目的 `answer_keys`（"主要讲什么"没有唯一答案，需人工给要点）
2. 所有题目的 `gold_chunk_ids`（现在是课程级近似判定，粒度偏松）
3. 真实用户问法——种子集是模板生成的，句式单一，**至少补 30 条口语化问句**，
   否则路由准确率会被高估

标注分工建议：每人 40 题，两人交叉复核，冲突记在 `notes` 里。

### E5 补充子集（问法多样、课名不带书名号）

模板题都带书名号、每类只有一种问法，测不出「两门课的关系」「有没有这门课」
「大几」这类真实问法的问题。E5 在种子集末尾追加五个子集（`subset` 字段区分，
只依赖 `courses.json`，relation 另读 concept 产物；追加在最后，既有子集的题
和随机序列不变）：

| 子集 | 题量 | 路由 | 构造与金标 |
| --- | --- | --- | --- |
| `relation-neg` | 15 | `dependency` | 随机课程对，排除直接先修、深度 ≤3 的间接先修、知识关联（两个方向都算）。三种问法：「X与Y之间有什么关系」「X和Y有关系吗」「《X》与《Y》有什么联系」。不可回答题，期望拒答（`error_code=COURSE_RELATION_NOT_FOUND`，文案含「无法说明两者存在关系」） |
| `relation-pos` | 15 | `dependency` | 有先修边或知识关联的课程对，直接 / 间接 / 知识关联各取约三分之一（某类不足由其他类补）；`answer_keys` 为两门课的课名，`notes` 记 `relation=direct|indirect|concept` |
| `existence` | 15 | `existence` | 10 门存在的课（其中 5 题带专业线索，如「信管专业有X这门课吗」）+ 5 门不存在的课；课名不带书名号。存在 → 答案以「有。」开头；带线索且该专业没有 → 含「中没有《X》」；课程不存在 → 拒答或反问 |
| `semester-phrasing` | 30 | `fact` | 10 门课 × 3 种问法（大几 / 什么时候上 / 这门课是大几才开 / 哪一年开）。只选各方案学期一致、且为单个整数学期的课，按学期轮流取 |
| `no-brackets` | 20 | 同源题 | 从既有 fact / dependency 题各抽 10 题，去掉书名号，路由与答案要点不变（`notes` 记源题 id） |

判定细节：

- 先修关系取 `courses[].prerequisites` 的有向图；知识关联复用
  `program_graph.aggregate_concept_links`，`min_confidence` 取
  `config.concept_verified_min_confidence`（与 `/programs/graph` 一致）。
- **concept 产物缺失时**（`concept_registry.json` / `concept_verified_edges.json`），
  relation 子集只按先修判定，并在 stderr 告警。此时 `relation-neg` 里可能混入
  实际有知识关联的课程对（`notes` 会标「concept 产物缺失」），不要拿这版金标
  去算正式数字。
- 关系子集不选课名含「与 / 和 / 及 / 、」的课（「X与Y」问法本身有歧义），也不选
  同名课、一个课名是另一个子串的课程对。
- 带专业线索的存在性题只在无歧义的两端出题：主修方案里有 → 「有」；该专业任何
  类型的方案里都没有 → 「没有」；只在辅修方案出现的课跳过。「信管」依赖部署侧的
  `PROGRAM_ALIASES`。
- 学期题用 `第\s*n\s*学期` 匹配，并同时检查年级换算（奇数上、偶数下，如第 4 学期
  对应「大二下」）。

用真实数据生成时用环境变量指定产物位置（`COURSES_OUTPUT_PATH`、`CHUNKS_OUTPUT_PATH`、
`CONCEPT_REGISTRY_PATH`、`CONCEPT_VERIFIED_EDGE_PATH`）。

---

## 2. 指标定义

### 路由

- `accuracy`：主标签命中率
- `label_recall`：多标签里包含正确路由即算命中——它与 `accuracy` 的差值就是
  多意图改造带来的收益
- `mean_confidence`：证据份额置信度均值（单意图恒为 1.0）
- 混淆矩阵：行是金标，列是预测

### 检索

- `recall@1/5/10`、`mrr`
- 判定规则见上文；`min(positions)` 用于 MRR

### 生成与引用

- `answer_key_coverage`：答案要点全命中率
- `citation_precision`：引用指向的 chunk/课程是否被金标接受
- `uncited_answer_rate`：给出答案却没有任何引用的比例（可溯性的反面指标）

### 答案切题率（E5）

只对 `existence`、`semester-phrasing` 两个子集计算，衡量「不该说的说了没有」：

- `answer_on_topic_rate`：答案第一句（按「。」或换行切分，保留句末「。」）是否含期望
  片段——`existence` 看 `focus_keys`（「有。」/「没有」或反问「你是不是想问」），
  `semester-phrasing` 看「第 n 学期」。答案里只要出现章节标题行（如
  「第一章 绪论（3 学时）」，`coverage_filler` 带进来的整门课章节），一律判不切题
- `chapter_heading_rate`：答案含章节标题行的比例
- `<子集>_on_topic_rate`：分子集的切题率
- `semester-phrasing` 的 `answer_correct` 另用正则 + 年级换算计分，见上

### 拒答

- `correct_refusal_rate`：无答案题里正确拒答的比例
- `hallucination_rate` = 1 − 正确拒答率（无答案题却给了答案）
- `false_refusal_rate`：有答案题里被误拒的比例

拒答判定基于 `metrics.REFUSAL_MARKERS`（含 S3 关系闸门的「无法说明两者存在关系」）——生成端改了拒答措辞，这里必须同步，
`tests/test_eval_harness.py` 会盯着这一点。

---

## 3. 实验脚本

| 脚本 | 产出 | 论文用途 |
| --- | --- | --- |
| `run_eval.py` | 主报告（四组指标） | 实验章节主表 |
| `ablation.py` | 各消融组 Recall/MRR 对比 | 证明重排、词面分、章节加权各自有效 |
| `tune_rerank.py` | 最优权重 + 敏感性曲线 | 回答"为什么是 0.75" |
| `persona_overlap.py` | 角色两两重合率 | 决定角色感知的表述强度 |
| `concept_eval.py` | 规则基线 vs LLM 抽取 P/R/F1 | 概念抽取质量表 |
| `concept_edges_eval.py` | 概念依赖边分层抽检与准确率 | 验证入选边的真实有效率 |

```bash
python eval/ablation.py --with-answers
python eval/tune_rerank.py --metric recall@5
python eval/persona_overlap.py --top-k 5
python eval/concept_eval.py --make-template --courses 20   # 先标注
python eval/concept_eval.py
python eval/concept_edges_eval.py --make-template           # 生成 50 条分层抽检
python eval/concept_edges_eval.py                           # 标注后计算准确率
```

`concept_edges_eval.py` 按有向课程对分层，从 `requires=true` 的入选边中
抽取 50 条；模板中标 `成立 / 不成立 / 勉强`。严格准确率只把“成立”算正确，
同时报告“成立+勉强”的可接受率、样本量和抽样方式。`run_eval.py` 检测到
`eval/datasets/concept_edges_gold.json` 时，也会把该指标写入主评测报告。

### 嵌入模型消融（hash vs 真实向量）

需要重建索引，因此单独跑：

```bash
EMBEDDING_PROVIDER=hash python run_pipeline.py --stage embed
python eval/run_eval.py --stages retrieval --tag hash-embedding
EMBEDDING_PROVIDER=local python run_pipeline.py --stage embed
python eval/run_eval.py --stages retrieval --tag local-embedding
```

### 反馈闭环的 before/after

```bash
python maintenance/active_learning_sampler.py     # 采样待复核
# 人工修正 -> outputs/corrected_samples.json
python maintenance/retraining_updater.py --evaluate
```

新版本建成 `zstt_chunks_v<n>` 后先在金标集上评测，指标不降才切换别名；
降了就保持原样，或 `--force` 强切，`--rollback` 回退。

---

## 4. 环境缺失时的行为

| 缺失 | 影响 |
| --- | --- |
| 无向量索引 | `retrieval` / `answer` 阶段跳过，`routing` 照常 |
| 无 `DEEPSEEK_API_KEY` | `answer` 阶段跑模板降级路径，报告标记 `llm_available: false` |
| 无 Neo4j | 依赖类问题走降级回答，报告标记 `neo4j_available: false` |

报告头部记录了 embedding 模型、collection 名、重排权重与后端可用性——
**跨版本比较前先核对这一段**，否则比的是两套环境。
