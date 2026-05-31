# Drainage 毕业设计项目

本项目为排水系统智能调度平台（Flask + MySQL + 前端可视化），包含：
- 后台数据查询与统计
- 仿真与决策演示
- 管理员/用户权限管理

## 1. 运行环境
- Python 3.10+
- MySQL 8+

## 2. 安装依赖
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2.1 运行基础测试
```bash
source .venv/bin/activate
pytest -q
```

## 3. 环境变量（必须）
建议在项目根目录创建 `.env` 并导出（或手动 export）：

快速方式：
```bash
cp .env.example .env
```
然后把 `.env` 里的 MySQL 用户名和密码改成你的实际值。

```bash
export FLASK_SECRET_KEY="replace-with-random-string"
export MYSQL_HOST="127.0.0.1"
export MYSQL_PORT="3306"
export MYSQL_USER="your_user"
export MYSQL_PASSWORD="your_password"
export MYSQL_DB="drainage"

# 生产/答辩建议
export FLASK_DEBUG="false"
export ALLOW_PUBLIC_REGISTER="false"

# 可选：管理员初始化保护令牌
export ADMIN_INIT_TOKEN="your_init_token"
```

加载 `.env` 后启动（zsh/bash）：
```bash
set -a
source .env
set +a
python app.py
```

## 4. 数据库准备
确保数据库中存在：
- `admin_users`（系统自动初始化）
- `inventory_data`（业务主表，查询/统计均基于此表）

## 5. 启动项目
```bash
source .venv/bin/activate
python app.py
```

一键启动（推荐）：
```bash
./run.sh
```

默认地址：`http://127.0.0.1:5000`

## 6. 论文演示建议流程
1. 登录后台（管理员）
2. 打开“查询筛选”，按时间与降雨/入流过滤
3. 展示“统计概览 + 趋势与分布”
4. 在记录管理中新增/编辑一条记录并刷新统计
5. 切换到仿真与决策页面展示联动

## 7. 说明
- 当前后台查询主表为 `inventory_data`。
- 风险等级和溢流为后端按水位规则推导，不依赖额外字段。
