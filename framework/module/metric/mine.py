from minepy import MINE


class MyMINE():
    def __init__(self):
        super(MyMINE, self).__init__()
        self.mine = MINE()

    def mic(self, x, y):
        self.mine.compute_score(x, y)
        return self.mine.mic()
