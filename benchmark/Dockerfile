# 可复现的基准运行环境。构建上下文必须是仓库根目录：
#   docker build -f benchmark/Dockerfile -t react-agent-bench .
#   docker run --rm -e DEEPSEEK_API_KEY react-agent-bench
#
# 这不是安全沙箱。它只固定 Python/依赖版本，便于比较不同机器上的结果；
# 真要隔离不可信代码，请再套一层 gVisor/firejail/虚拟机或远程执行服务。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FILE_ACCESS_MODE=root \
    FILE_ROOT=/app \
    AGENT_MEMORY_FILE=/tmp/agent-memory.json

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --uid 10001 bench \
    && mkdir -p /app/benchmark/results \
    && chown -R bench:bench /app
USER bench

# 默认跑全部任务；也可以覆盖为 --reference / 单个任务 / metrics.py。
CMD ["python", "-X", "utf8", "benchmark/runner.py", "--all"]
