# data/ —— 语料目录（不进仓库）

这里放教学大纲与培养方案。**目录内容被 `.gitignore` 排除**：原始文件含任课教师
姓名、教材、考核方式等信息，属于院系资产，不应随公开仓库分发。

```
data/
├── syllabi/         教学大纲，*.docx（可任意分子目录，递归扫描）
└── training_plans/  培养方案，*.xlsx
```

## 放进去的三种方式

**1. 直接拷贝（本地开发最快）**

```bash
cp 你的大纲/*.docx data/syllabi/
cp 你的培养方案/*.xlsx data/training_plans/
python run_pipeline.py --stage all
```

**2. 导入接口（部署后用这个）**

需要先设置 `ADMIN_TOKEN`，否则 `/admin/*` 全部返回 503。

```bash
curl -X POST http://127.0.0.1:8000/admin/data/import \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -F "files=@IM104档案学概论.docx" \
  -F "files=@25级培养方案.xlsx"

curl -X POST http://127.0.0.1:8000/admin/data/reindex \
  -H "Authorization: Bearer $ADMIN_TOKEN"          # 后台重建，立即返回

curl http://127.0.0.1:8000/admin/data/reindex \
  -H "Authorization: Bearer $ADMIN_TOKEN"          # 轮询状态
curl http://127.0.0.1:8000/admin/data \
  -H "Authorization: Bearer $ADMIN_TOKEN"          # 看当前语料清单
```

只接受 `.docx` 与 `.xlsx`，单文件上限由 `IMPORT_MAX_BYTES` 控制（默认 20MB）。
文件名会被去掉路径成分，`~$` 开头的 Office 临时文件直接拒绝。

**3. 容器部署**

`data` 是命名卷 `corpus_data`，批量灌入用：

```bash
docker compose cp ./你的语料/. api:/app/data/
docker compose --profile pipeline run --rm pipeline
```

## 换一所学校的培养方案要改什么

代码里没有任何学校或专业名。专业名与方案类型（主修/辅修/辅修微专业……）都从
解析结果 `outputs/courses.json` 里自动学习，xlsx 表头也是模糊匹配
（`课程编码/课程代码/课程编号`、`课程中文名称/课程名称`…）。

唯一需要配的是**口语缩写**，因为它不出现在任何方案标题里：

```dotenv
PROGRAM_ALIASES=信管=信息管理与信息系统,图情=图书情报与档案管理类
```

不配也能用，只是学生打"信管"时匹配不到专业，得打全称。
