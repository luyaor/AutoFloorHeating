from typing import Dict, List


class DisjointSet:
    def __init__(self):
        self.fa_di = dict()

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
        return {x: self.find(x) for x in self.fa_di}


# test
def test_disjoint_set():
    ds = DisjointSet()
    ds.add(1)
    ds.add(2)
    assert ds.find(1) == 1
    ds.mix(1, 2)
    assert ds.find(1) == 2

    ds.add(3)
    ds.add(4)
    ds.add(5)
    ds.mix(3, 4)
    assert ds.find(3) == 4
    ds.mix(1, 4)
    assert ds.find(1) == 4
    assert ds.find(2) == 4
    assert ds.find(3) == 4
    assert ds.find(5) == 5
    assert ds.get_sets_di() == {4: [1, 2, 3, 4], 5: [5]}


if __name__ == "__main__":
    test_disjoint_set()
