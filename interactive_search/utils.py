def get_multiline_input() -> str:
    print(" Введите фрагмент кода для поиска.")
    print(
        "   (Для запуска поиска введите 'RUN' на новой строке. Для выхода введите 'EXIT'):"
    )
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == "EXIT":
                return "EXIT"
            if line.strip().upper() == "RUN":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)
