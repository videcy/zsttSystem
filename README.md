# zsttSystem 本地部署说明

## 项目说明

本项目用于本地演示课程培养方案与教学大纲的智能问答系统，整体流程分为两个阶段：

- 离线构建：解析培养方案和课程大纲，生成分块数据、向量索引和知识抽取结果
- 在线服务：启动本地 FastAPI 接口，提供问答与反馈功能

当前仓库已经按“本地可运行、可展示”的目标调整，不再依赖云端部署流程。

## 运行环境

建议使用以下环境：

- Windows 10 或 Windows 11
- Python 3.11
- PowerShell
- 可用的 GLM/OpenAI 兼容接口密钥
- 可选：本地或远程 Neo4j

说明：

- 即使 Neo4j 当前不可用，系统也可以用本地降级模式完成离线构建和基础问答展示
- Neo4j 可用时，会启用图谱相关能力

## 项目结构

- `run_pipeline.py`：离线流程入口
- `src/main.py`：在线服务入口
- `data/training_plans`：培养方案 Excel
- `data/syllabi`：课程大纲 Word 文档
- `outputs`：离线处理结果和日志
- `vector_store`：本地向量库文件

## 第一步：创建虚拟环境

在项目根目录执行：

```powershell
cd D:\python\zsttSystem
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 提示脚本执行被禁止，先执行：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

然后重新执行：

```powershell
.\.venv\Scripts\Activate.ps1
```

## 第二步：安装依赖

激活虚拟环境后执行：

```powershell
pip install -r requirements.txt
```

## 第三步：填写 `.env`

本项目直接使用根目录下的 `.env` 文件。

如果项目根目录下还没有 `.env`，请手动新建该文件，并填写类似下面的内容：

```env
ZAI_API_KEY=你的接口密钥
ZAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
GLM_TEXT_MODEL=glm-5
GLM_EMBEDDING_MODEL=embedding-3
GLM_RERANK_MODEL=glm-5
GLM_JUDGE_MODEL=glm-5

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=你的密码

VECTOR_DB_PATH=vector_store
QUERY_LOG_PATH=outputs/query_log.jsonl
FEEDBACK_LOG_PATH=outputs/feedback_log.jsonl
TRAINING_PLAN_DIR=data/training_plans
SYLLABUS_DIR=data/syllabi
CHUNKED_OUTPUT_PATH=outputs/chunked_data.json
KG_OUTPUT_PATH=outputs/kg_extracted_data.json
```

说明：

- 如果暂时没有可用的 Neo4j，可以先保留 Neo4j 配置，系统会在部分流程中自动降级
- 如果你的模型名在接口侧不可用，可以后续再调整 `GLM_TEXT_MODEL`、`GLM_RERANK_MODEL`、`GLM_JUDGE_MODEL`

## 第四步：运行离线构建

推荐直接执行：

```powershell
.\run_local_pipeline.ps1
```

它会顺序运行以下阶段：

- `parsing`
- `vectorization`
- `kg`
- `alignment`

如果你要单独调试某一阶段，也可以执行：

```powershell
python run_pipeline.py --stage parsing
python run_pipeline.py --stage vectorization
python run_pipeline.py --stage kg
python run_pipeline.py --stage alignment
```

离线构建成功后，常见输出文件包括：

- `outputs/chunked_data.json`
- `outputs/kg_extracted_data.json`
- `vector_store/scholar_collection.json`

## 第五步：启动本地服务

推荐执行：

```powershell
.\start_local_api.ps1
```

等价命令为：

```powershell
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```

## 第六步：打开演示页面

服务启动后，可以直接在浏览器访问：

- `http://127.0.0.1:8000/`：本地演示页
- `http://127.0.0.1:8000/health`：健康检查接口

可用接口如下：

- `GET /`
- `GET /health`
- `POST /query`
- `POST /feedback`

## 常见问题

### 1. 培养方案缺少课程代码和课程名称

这个问题通常来自 Excel 表头不在第一行。当前代码已经兼容扫描课程表头，如果再次出现，优先检查新增 Excel 的表格结构是否异常。

### 2. embedding 接口报批量上限或参数错误

当前代码已经做了两层处理：

- embedding 请求自动分批
- 超长文本自动截断

如果仍然报错，优先检查 `.env` 中的模型名和接口密钥。

### 3. Neo4j 认证失败

如果 Neo4j 用户名、密码或地址错误，图谱相关能力会受影响，但当前本地演示流程已经支持降级运行，不会直接阻塞基础问答展示。

### 4. 问答接口返回 500

优先检查：

- 服务是否已经重启到最新代码
- `.env` 是否填写完整
- 离线构建是否执行成功
- `outputs/chunked_data.json` 和 `vector_store/scholar_collection.json` 是否已经生成

## 建议的本地演示顺序

1. 激活虚拟环境
2. 检查 `.env`
3. 运行 `.\run_local_pipeline.ps1`
4. 运行 `.\start_local_api.ps1`
5. 打开 `http://127.0.0.1:8000/`
6. 输入问题进行展示

## 注意事项

- 不要提交真实 `.env`
- `outputs/` 和 `vector_store/` 会随着本地构建不断更新
- 如果你修改了代码，重新启动 `uvicorn` 后新逻辑才会生效
