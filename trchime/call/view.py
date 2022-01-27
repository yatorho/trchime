from typing import List


class ProgressBar:
    """
    Show the process for training
    """

    def __init__(self, width=50):
        self.pointer = 0

        self.width = width

    def __call__(self, x):
        # x in percent
        assert 100 >= x >= 0, "progressbar called argument should be in range(0, 101)"

        self.pointer = int(self.width * (x / 100.0))

        return "|" + "#" * self.pointer + "-" * (self.width - self.pointer) + "| %d %% done" % int(x)


class MessageBoard:

    def __init__(self, width: int = 50):
        self.width = width
        self.total_lines = 0
        self.lines: List[_line] = []

    def add_horizontal_line(self, full: int = 0):
        self.lines.append(_line("", self.total_lines))

        if full == 1:
            for _ in range(self.width):
                self.lines[self.total_lines].write("-")

        elif full == 2:
            for _ in range(self.width):
                self.lines[self.total_lines].write("=")

        self.total_lines += 1


    def add_text(self, line: int, text: str, width: int = 10, sperarator: bool = False):
        if sperarator:
            string = "%-{0}s|".format(width) % text
        else:
            string = "%-{0}s".format(width) % text
        self.lines[line - 1].write(string)

    def show(self):
        for line in self.lines:
            line.delete(1)
            line.show()


class _line:
    def __init__(self, text, line: int):
        self.line_nums = line
        self.text: str = text

    def write(self, text: str):
        self.text += text
        return len(self.text)

    def delete(self, length: int, from_head: bool = False):
        if from_head:
            self.text = self.text[length:]
        else:
            self.text = self.text[:len(self.text) - length]
        return len(self.text)

    def show(self):
        print(self.text)
