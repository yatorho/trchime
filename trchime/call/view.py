

class ProgressBar:
    """
    Show the process for traning
    """
    def __init__(self, width=50):

        self.pointer = 0

        self.width = width

    def __call__(self, x):
        # x in percent
        assert 100 >= x >= 0, "progressbar called argument should be in range(0, 101)"

        self.pointer = int(self.width*(x/100.0))

        return "|" + "#"*self.pointer + "-"*(self.width-self.pointer)+ "| %d %% done" % int(x)

class MessageBoard:
    """

    """

    def __init__(self, height=30, width=50):
        self.height = height
        self.width = width

    def addhorizontalline(self):
        pass




