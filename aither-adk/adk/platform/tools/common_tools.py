import sys
import time

class ProgressBar:
    """
    A simple text-based progress bar.
    """
    def __init__(self, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd="\r"):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.printEnd = printEnd
        self.iteration = 0

    def update(self, iteration, suffix=None):
        """
        Update the progress bar to a specific iteration.
        """
        self.iteration = iteration
        if suffix:
            self.suffix = suffix
        self.print()

    def increment(self, amount=1, suffix=None):
        """
        Increment the progress bar by a specific amount.
        """
        self.update(self.iteration + amount, suffix)

    def print(self):
        """
        Print the current state of the progress bar.
        """
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.iteration / float(self.total)))
        filledLength = int(self.length * self.iteration // self.total)
        bar = self.fill * filledLength + '-' * (self.length - filledLength)
        print(f'\r{self.prefix} |{bar}| {percent}% {self.suffix}', end=self.printEnd)
        # Print New Line on Complete
        if self.iteration == self.total:
            print()
