"""极简 CSV 解析。"""


def parse_line(line):
    fields = []
    current = []
    in_quotes = False
    index = 0
    while index < len(line):
        char = line[index]
        if in_quotes:
            if char == '"':
                if index + 1 < len(line) and line[index + 1] == '"':
                    current.append('"')
                    index += 1
                else:
                    in_quotes = False
            else:
                current.append(char)
        else:
            if char == '"':
                in_quotes = True
            elif char == ",":
                fields.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        index += 1
    if in_quotes:
        raise ValueError("CSV 字段引号未闭合")
    fields.append("".join(current).strip())
    return fields


def parse_csv(text):
    return [parse_line(line) for line in text.splitlines() if line.strip()]
