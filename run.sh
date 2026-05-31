#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d ".venv" ]]; then
  echo "未找到 .venv，请先创建虚拟环境：python3 -m venv .venv"
  exit 1
fi

if [[ ! -f ".env" ]]; then
  if [[ -f ".env.example" ]]; then
    cp .env.example .env
    echo "已自动创建 .env，请先编辑其中的 MYSQL_PASSWORD 后再运行。"
  else
    echo "未找到 .env，请先创建并配置数据库连接。"
  fi
  exit 1
fi

source .venv/bin/activate
set -a
source .env
set +a

exec python app.py
