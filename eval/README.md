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

- `expected_route`：`fact` / `content` / `dependency` / `catalog` / `hybrid`
- `answer_keys`：**全部命中才算答对**；单个 key 内用 `|` 表示"任一等价说法"
- `gold_chunk_ids`：人工标注的证据 chunk；**有它就按 chunk 级判定**
- `gold_course_codes`：没有 chunk 标注时的课程级近似判定（报告会分别统计
  `chunk_level_items` / `course_level_items`，写论文时必须说明比例）
- `answerable: false`：无答案题，用于测拒答；这类题**不允许**有 `answer_keys`

### 种子集自动生成的边界

自动生成能覆盖的：培养方案里已有确定值的字段题（学分/学时/学期/教师）、
先修关系题、培养方案目录题、无答案题。

**必须人工补的**：

1. `content` 类题目的 `answer_keys`（"主要讲什么"没有唯一答案，需人工给要点）
2. 所有题目的 `gold_chunk_ids`（现在是课程级近似判定，粒度偏松）
3. 真实用户问法——种子集是模板生成的，句式单一，**至少补 30 条口语化问句**，
   否则路由准确率会被高估

标注分工建议：每人 40 题，两人交叉复核，冲突记在 `notes` 里。

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

### 拒答

- `correct_refusal_rate`：无答案题里正确拒答的比例
- `hallucination_rate` = 1 − 正确拒答率（无答案题却给了答案）
- `false_refusal_rate`：有答案题里被误拒的比例

拒答判定基于 `metrics.REFUSAL_MARKERS`——生成端改了拒答措辞，这里必须同步，
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

```bash
python eval/ablation.py --with-answers
python eval/tune_rerank.py --metric recall@5
python eval/persona_overlap.py --top-k 5
python eval/concept_eval.py --make-template --courses 20   # 先标注
python eval/concept_eval.py
```

### 嵌入模型消融（hash vs 真实向量）

需要重建索引，因此单独跑：

```bash
EMBEDDING_PROVIDER=hash python run_pipeline.py embed
python eval/run_eval.py --stages retrieval --tag hash-embedding
EMBEDDING_PROVIDER=local python run_pipeline.py embed
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
