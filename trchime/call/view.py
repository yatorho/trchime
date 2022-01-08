

class ProgressBar:
    """
    Show the process for traning
    """
    def __init__(self, width=50):

        self.pointer = 0

        self.width = width

    def __call__(self, x):
        # x in percent
        assert 100 >= x >= 0, "progressbar called argment should be in range(0, 101)"

        self.pointer = int(self.width*(x/100.0))

        return "|" + "#"*self.pointer + "-"*(self.width-self.pointer)+ "| %d %% done" % int(x)



