"""极简 CSV 解析。"""


def parse_line(line):
    return [part.strip() for part in line.split(",")]


def parse_csv(text):
    return [parse_line(line) for line in text.splitlines() if line.strip()]
