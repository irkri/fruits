class ExtendedLetter(list):
    def __init__(self, *letters):
        super().__init__(letters)

    def __repr__(self):
        return "fruits.core.letters.ExtendedLetter"
