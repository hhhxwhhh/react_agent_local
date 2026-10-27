"""配置加载：优先读取环境变量，其次读取同目录下的 .env 文件。"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"


def load_env_file(path: Path = ENV_FILE) -> None:
    """极简 .env 读取器，避免引入 python-dotenv 依赖。"""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # 已存在的环境变量优先，方便临时覆盖 .env
        os.environ.setdefault(key, value)


load_env_file()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "90"))
# 单次模型调用的输出上限。1024 容易把较长的最终答案截断，默认放宽到 2048。
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "2048"))
# Web 服务同时运行的推理任务上限，避免多个会话各自起线程把本机资源占满。
MAX_CONCURRENT_RUNS = max(1, int(os.getenv("MAX_CONCURRENT_RUNS", "3")))
# 调用 get_location 时等待用户在界面上授权浏览器定位的时长；超时回退到 IP 定位。
LOCATION_PERMISSION_TIMEOUT = max(5, int(os.getenv("LOCATION_PERMISSION_TIMEOUT", "90")))

# 文件访问范围：unrestricted 允许本机任意绝对路径；workspace 恢复旧的文件沙箱。
FILE_ACCESS_MODE = os.getenv("FILE_ACCESS_MODE", "unrestricted").strip().lower()
if FILE_ACCESS_MODE not in {"unrestricted", "workspace"}:
    FILE_ACCESS_MODE = "unrestricted"
# 相对路径的基准目录；默认项目根目录。
FILE_ROOT = Path(os.getenv("FILE_ROOT", str(BASE_DIR))).expanduser().resolve()

# 写文件、编辑文件、执行命令前是否请求用户确认。
REQUIRE_TOOL_APPROVAL = os.getenv(
    "REQUIRE_TOOL_APPROVAL", "true"
).strip().lower() not in {"0", "false", "off", "no"}
TOOL_PERMISSION_TIMEOUT = max(10, int(os.getenv("TOOL_PERMISSION_TIMEOUT", "120")))

# 本机命令执行：默认超时与输出上限。
COMMAND_TIMEOUT = max(5, int(os.getenv("COMMAND_TIMEOUT", "120")))
COMMAND_MAX_OUTPUT_CHARS = max(1000, int(os.getenv("COMMAND_MAX_OUTPUT_CHARS", "12000")))

# 自进化 Harness：单代改动预算、测试超时和 Agent 最大工具轮数。
EVOLUTION_MAX_DIFF_LINES = max(10, int(os.getenv("EVOLUTION_MAX_DIFF_LINES", "600")))
EVOLUTION_TEST_TIMEOUT = max(30, int(os.getenv("EVOLUTION_TEST_TIMEOUT", "300")))
EVOLUTION_MAX_STEPS = max(5, int(os.getenv("EVOLUTION_MAX_STEPS", "40")))

# 单轮 ReAct 上下文压缩：scratchpad 超过阈值后，把较早步骤摘要掉，只保留最近原文。
CONTEXT_COMPACT_CHARS = max(800, int(os.getenv("CONTEXT_COMPACT_CHARS", "7000")))
CONTEXT_KEEP_CHARS = max(300, int(os.getenv("CONTEXT_KEEP_CHARS", "2400")))
CONTEXT_SUMMARY_MAX_CHARS = max(300, int(os.getenv("CONTEXT_SUMMARY_MAX_CHARS", "1200")))

# 跨轮会话历史压缩：历史过长时，把最近窗口之外的对话增量摘要并持久化。
HISTORY_COMPACT_CHARS = max(2000, int(os.getenv("HISTORY_COMPACT_CHARS", "9000")))
HISTORY_COMPACT_MESSAGES = max(6, int(os.getenv("HISTORY_COMPACT_MESSAGES", "12")))
HISTORY_KEEP_MESSAGES = max(2, int(os.getenv("HISTORY_KEEP_MESSAGES", "8")))
HISTORY_SUMMARY_BATCH_MESSAGES = max(
    1, int(os.getenv("HISTORY_SUMMARY_BATCH_MESSAGES", "4"))
)

# 向量记忆索引。没有额外 embedding API 时使用本地特征哈希，零依赖、可离线。
# 可设为 false 做“向量检索 vs 纯关键词”的消融实验。
VECTOR_MEMORY_ENABLED = os.getenv("VECTOR_MEMORY_ENABLED", "true").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
}
VECTOR_DIM = max(64, int(os.getenv("VECTOR_DIM", "512")))
VECTOR_SEARCH_LIMIT = max(1, min(20, int(os.getenv("VECTOR_SEARCH_LIMIT", "5"))))
VECTOR_MIN_SCORE = min(1.0, max(0.0, float(os.getenv("VECTOR_MIN_SCORE", "0.05"))))
# 混合检索中余弦相似度的权重；剩余权重给字面重合分数。
VECTOR_HYBRID_ALPHA = min(1.0, max(0.0, float(os.getenv("VECTOR_HYBRID_ALPHA", "0.7"))))
# 记忆条目超过这个数量时，注入提示词前按与问题的相关性重排。
SEMANTIC_RERANK_MIN_ENTRIES = max(2, int(os.getenv("SEMANTIC_RERANK_MIN_ENTRIES", "6")))
