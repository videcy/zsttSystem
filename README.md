# zsttSystem v2.0 本地部署说明

## 项目说明

本项目是一个面向高校课程教学大纲与培养方案的智能问答系统，采用"领域知识层 + LightRAG 检索引擎层"双层协作架构：

- **zsttSystem 领域知识层**：负责教学大纲解析、概念标准化、知识图谱构建（Neo4j）、课程依赖推理，以及 HyDE 查询扩展和 NLI 事实验证
- **LightRAG 检索引擎层**：提供文本索引、多模式检索（5 种）、结果重排、答案生成和 LLM 响应缓存

两层通过 HTTP API 协作，互不侵入，各自独立演进。

## 运行环境

- Windows 10 / 11（Linux 和 macOS 亦可）
- Python 3.11+
- 可用的 DeepSeek API 密钥（或其他 OpenAI 兼容 API）
- LightRAG v1.5+
- 可选：本地或远程 Neo4j（依赖查询和知识图谱构建需要）

## 项目结构

```
zsttSystem/
├── run_pipeline.py              # 离线管线入口（含 sync 阶段）
├── src/
│   ├── main.py                  # FastAPI 在线服务入口
│   ├── config.py                # 统一配置管理
│   ├── data_processing/         # 离线管线模块
│   │   ├── parser_chunker.py    #   教学大纲解析
│   │   ├── kg_builder.py        #   知识图谱构建
│   │   ├── aligner.py           #   KG 节点元数据对齐
│   │   ├── module_dependency.py #   课程依赖聚合
│   │   └── data_bridge.py       #   LightRAG 数据同步
│   ├── online_service/          # 在线服务模块
│   │   ├── query_router.py      #   查询路由（意图分类 + HyDE + NLI）
│   │   ├── lightrag_adapter.py  #   LightRAG API 客户端
│   │   ├── dependency_explainer.py  #   依赖解释
│   │   ├── generator.py         #   NLI 事实验证
│   │   └── feedback_handler.py  #   反馈日志
│   └── utils/
├── data/                        # 原始数据
│   ├── training_plans/          #   培养方案（XLSX）
│   └── syllabi/                 #   教学大纲（DOCX）
├── outputs/                     # 管线产出
├── .env.example                 # 环境变量模板
└── requirements.txt             # Python 依赖
```

## 第一步：创建虚拟环境

```powershell
cd D:\python\zsttSystem1.1\zsttSystem
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 提示脚本执行被禁止：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## 第二步：安装依赖

```powershell
pip install -r requirements.txt
pip install "lightrag-hku[api]"
```

## 第三步：配置环境变量

```powershell
copy .env.example .env
```

编辑 `.env`，填入必要信息：

```env
# DeepSeek API
DEEPSEEK_API_KEY=sk-你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
TEXT_MODEL=deepseek-v4-flash

# Neo4j（可选，依赖查询需要）
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=你的密码

# LightRAG
LIGHTRAG_BASE_URL=http://127.0.0.1:9621
DEFAULT_QUERY_MODE=mix
ENABLE_HYDE_EXPANSION=true
ENABLE_NLI_VERIFICATION=false
ENABLE_CONCEPT_NORMALIZATION=true
```

完整的配置项说明见 `.env.example`。

## 第四步：启动基础设施

### 一键启动（推荐）

双击 `start_all.bat`，或在 PowerShell 中执行：

```powershell
cd D:\python\zsttSystem1.1\zsttSystem
.\start_all.bat
```

该脚本会依次启动：Embedding 服务（端口 11435）→ LightRAG（端口 9621）→ zsttSystem API（端口 8000）。

### 手动分步启动

如需分别启动各个服务：

**终端 1** — Embedding 服务（本地确定性嵌入，无需下载模型）：
```powershell
.venv\Scripts\Activate.ps1
python -m src.utils.embedding_server --port 11435
```

**终端 2** — LightRAG 检索引擎：
```powershell
.venv\Scripts\Activate.ps1
set EMBEDDING_DIM=384
python run_lightrag.py --port 9621 --llm-binding openai --key zstt_local_dev_key
```

看到 `Uvicorn running on http://0.0.0.0:9621` 表示启动成功。

## 第五步：运行离线管线

打开终端 2：

```powershell
.\.venv\Scripts\Activate.ps1
python run_pipeline.py --stage all
```

管线会依次执行：
1. `parsing`——解析教学大纲和培养方案
2. `kg`——知识图谱构建（需要 Neo4j）
3. `alignment`——KG 节点元数据对齐
4. `module`——课程依赖聚合
5. `sync`——同步数据到 LightRAG

如需增量运行（只处理新增/变更的大纲文件）：

```powershell
python run_pipeline.py --incremental
```

如需单独执行某一阶段：

```powershell
python run_pipeline.py --stage parsing     # 仅解析
python run_pipeline.py --stage sync        # 仅同步到 LightRAG
```

## 第六步：启动 zsttSystem API

如果未使用 `start_all.bat`，需单独启动：

```powershell
uvicorn src.main:app --host 127.0.0.1 --port 8000
```

## 第七步：验证

浏览器访问：

- `http://127.0.0.1:8000/`——演示页面
- `http://127.0.0.1:8000/health`——健康检查（返回 LightRAG 和 Neo4j 连接状态）

### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 演示页面 |
| GET | `/health` | 健康检查 |
| POST | `/query` | 主问答接口（自动路由） |
| GET | `/dependency?query=...` | 课程依赖查询 |
| POST | `/feedback` | 用户反馈 |

### 查询示例

```powershell
# 复杂问答（自动走 HyDE + LightRAG mix）
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "信息组织课程主要讲什么内容？"}'

# 依赖查询（走 Neo4j）
curl "http://127.0.0.1:8000/dependency?query=数据库原理需要哪些先修课程"

# 简单查询（走 LightRAG naive）
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "数据库原理多少学分？"}'
```

三种查询路径由 QueryRouter 根据意图自动分发，无需手动指定。

## 常见问题

### 1. LightRAG 连接失败

检查 `http://127.0.0.1:9621/health` 是否可达。如果 LightRAG 未启动，简单查询和复杂问答会返回 fallback 响应，依赖查询（Neo4j）不受影响。

### 2. Neo4j 认证失败

Neo4j 不可用时，知识图谱构建和依赖查询功能不可用，但基础的文本索引和检索问答（LightRAG naive 模式）可以正常工作。

### 3. 管线执行中断

支持断点续跑。例如 kg 阶段失败，修复问题后从该阶段继续：

```powershell
python run_pipeline.py --stage kg
python run_pipeline.py --stage alignment
python run_pipeline.py --stage module
python run_pipeline.py --stage sync
```

### 4. 增量更新

新增或修改大纲后，使用 `--incremental` 模式只处理变更文件：

```powershell
python run_pipeline.py --stage parsing --incremental
python run_pipeline.py --stage sync
```

也可全量重新处理：

```powershell
python run_pipeline.py --stage parsing --force
```

增量模式通过 SHA256 文件哈希检测变更，manifest 存储在 `outputs/.file_manifest.json`。

## 建议的演示顺序

1. 激活虚拟环境
2. 检查 `.env`
3. 终端 1：`lightrag-server`
4. 终端 2：`python run_pipeline.py --stage all`
5. 终端 2：`uvicorn src.main:app --host 127.0.0.1 --port 8000`
6. 浏览器打开 `http://127.0.0.1:8000/`，输入问题展示

## 注意事项

- 不要提交 `.env` 到版本控制
- 新增或修改大纲后，使用 `--incremental` 增量同步；用 `--force` 强制全量重跑
- 修改代码后重启 uvicorn 新逻辑才会生效
- LightRAG 依赖的 `.env` 变量（LLM_BINDING、EMBEDDING_BINDING 等）与 zsttSystem 共用一个 `.env` 文件
