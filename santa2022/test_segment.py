import unittest

from .segment import UnionFind

def max_area_of_island(grid):
    grid_map = {}
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == 1:
                grid_map[(i, j)] = cell

    uf = UnionFind(grid_map)
    for x in grid_map:
        if grid_map.get(x, 0) == 0:
            continue
        for d in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            y = (x[0] + d[0], x[1] + d[1])
            if grid_map.get(y, 0) == 1:
                uf.union(x, y)
    return max(uf.size.values(), default=0)


class UnionFindTestCase(unittest.TestCase):
    def test_leetcode(self):
        # https://leetcode.com/problems/max-area-of-island/

        grid = [
            [0,0,1,0,0,0,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,1,0,0,0],
            [0,1,1,0,1,0,0,0,0,0,0,0,0],
            [0,1,0,0,1,1,0,0,1,0,1,0,0],
            [0,1,0,0,1,1,0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,1,1,1,0,0,0],
            [0,0,0,0,0,0,0,1,1,0,0,0,0]
        ]
        self.assertEqual(max_area_of_island(grid), 6)
    