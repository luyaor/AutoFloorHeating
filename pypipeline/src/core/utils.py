from typing import Dict, List


class DisjointSet:
    def __init__(self):
        self.fa_di = dict()
        ...

    def add(self, *xs):
        for x in xs:
            if x not in self.fa_di:
                self.fa_di[x] = x

    def find(self, x):
        if self.fa_di[x] != x:
            self.fa_di[x] = self.find(self.fa_di[x])
        return self.fa_di[x]

    def mix(self, x, y):
        self.fa_di[self.find(x)] = self.find(y)

    def get_sets_di(self) -> Dict[int, List[int]]:
        """
        anc -> its set
        """
        di = {}
        for x in self.fa_di:
            fx = self.find(x)
            if fx not in di:
                di[fx] = []
            di[fx].append(x)
        return di

    def get_ancestor_di(self):
        return { x: self.find(x) for x in self.fa_di }