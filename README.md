# zsttSystem

面向高校培养方案与课程教学大纲的 RAG-KG 问答系统。系统使用：

- FastAPI 提供查询、课程和反馈 API。
- ChromaDB 保存课程文本向量并执行语义检索。
- Neo4j 提供可选的课程依赖图查询。
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
    │                    └─────────── Neo4j（可选）
    ▼
FastAPI → QueryRouter → ChromaDB / Neo4j → DeepSeek
```

## 运行要求

- Python 3.11 或 3.12
- DeepSeek API 密钥
- Docker Desktop（推荐，用于启动 ChromaDB）
- Neo4j（可选，仅依赖关系查询需要）

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

编辑 `.env`，至少设置：

```env
DEEPSEEK_API_KEY=your-deepseek-api-key
CHROMA_MODE=http
CHROMA_HOST=127.0.0.1
CHROMA_PORT=8001
```

### 2. 启动 ChromaDB

```bash
docker compose up -d chromadb
```

服务仅监听宿主机 `127.0.0.1:8001`，数据保存在 Docker 命名卷
`chroma_data` 中。

检查容器状态：

```bash
docker compose ps
```

### 3. 构建数据

```bash
python run_pipeline.py --stage all
```

`all` 会依次执行：

1. `parse`：解析培养方案和教学大纲。
2. `concept`：提取课程级概念。
3. `graph`：生成图摘要，并在 Neo4j 可用时写入图数据。
4. `embed`：生成向量并写入 ChromaDB 的 `zstt_chunks` collection。

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

```bash
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Windows 也可以运行：

```powershell
.\start_all.bat
```

该脚本会从脚本所在目录启动 ChromaDB 和 FastAPI，不依赖本机绝对路径。

### 5. 验证

- 演示页：`http://127.0.0.1:8000/`
- OpenAPI：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

查询示例：

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"数据库原理主要讲什么？\"}"
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
| GET | `/dependency?query=...` | 课程依赖查询 |
| POST | `/feedback` | 记录用户反馈 |

## 查询路由

- `fact`：优先匹配课程结构化信息，然后查询 ChromaDB。
- `content`：从 ChromaDB 返回最相关课程片段。
- `dependency`：查询 Neo4j；Neo4j 不可用时返回降级响应。
- `hybrid`：组合 ChromaDB 证据、Neo4j 路径和 LLM 生成。

LLM 暂时不可用时，hybrid 查询会退化为返回检索证据，不会直接产生
HTTP 500。

## 维护反馈数据

生成待审核样本：

```bash
python maintenance/active_learning_sampler.py
```

应用人工修正：

```bash
python maintenance/retraining_updater.py
```

修正课程片段后，维护脚本会更新本地 JSON，并重建当前配置的 ChromaDB
collection。

## 开发与测试

```bash
python -m pip install -r requirements-dev.txt
ruff check --select E9,F63,F7,F82 .
python -m compileall -q src maintenance tests run_pipeline.py
pytest -q
```

GitHub Actions 会在 Python 3.11 和 3.12 上执行相同检查。

## 常见问题

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

内容检索仍可正常使用。只有 dependency 查询和 hybrid 图路径会降级。

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
