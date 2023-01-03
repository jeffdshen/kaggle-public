import numpy as np
from collections import defaultdict


class UnionFind:
    def __init__(self, nodes):
        super().__init__()
        self.parent = {u: u for u in nodes}
        self.size = {u: 1 for u in nodes}
        self.num = len(nodes)

    def find(self, u):
        root = u
        while self.parent[root] != root:
            root = self.parent[root]

        while self.parent[u] != root:
            parent = self.parent[u]
            self.parent[u] = root
            u = parent

        return root

    def union(self, u, v):
        u = self.find(u)
        v = self.find(v)
        if u == v:
            return False

        if self.size[u] < self.size[v]:
            u, v = v, u

        self.parent[v] = u
        self.size[u] += self.size[v]
        self.num -= 1
        return True


def get_pixel_cost(color_a, color_b, u, v, color_weight=3.0):
    return color_weight * np.abs(np.subtract(color_a, color_b)).sum() + np.sqrt(
        np.sum(np.abs(np.subtract(u, v)))
    )


def combine_boxes(b1, b2):
    xmin1, xmax1, ymin1, ymax1 = b1
    xmin2, xmax2, ymin2, ymax2 = b2
    return min(xmin1, xmin2), max(xmax1, xmax2), min(ymin1, ymin2), max(ymax1, ymax2)


def check_box(box, size):
    xmin, xmax, ymin, ymax = box
    return xmax - xmin <= size and ymax - ymin <= size


def in_box(u, box):
    xmin, xmax, ymin, ymax = box
    x, y = u
    return xmin <= x < xmax and ymin <= y < ymax


def sum_points(u, d):
    return (u[0] + d[0], u[1] + d[1])


def run_segmentation(
    image_map,
    max_distance,
    min_count=32,
    max_size=64,
    neighbors=[(0, 1), (1, 0), (-1, 0), (0, -1)],
):
    uf = UnionFind(image_map)
    edges = []
    component_edges = []
    for u in image_map:
        a = image_map[u]
        for d in neighbors:
            v = sum_points(u, d)
            if v not in image_map:
                continue
            if v < u:
                continue
            b = image_map[v]
            edges.append((get_pixel_cost(a, b, u, v), (u, v)))

    edges.sort()

    boxes = {(x, y): (x, x + 1, y, y + 1) for x, y in image_map}
    for cost, edge in edges:
        u, v = edge
        u, v = uf.find(u), uf.find(v)
        if cost > max_distance and uf.size[u] >= min_count and uf.size[v] >= min_count:
            continue
        box = combine_boxes(boxes[u], boxes[v])
        if not check_box(box, max_size):
            continue
        if not uf.union(u, v):
            continue

        u = uf.find(u)
        boxes[u] = box
        component_edges.append(edge)
    return uf, component_edges


def get_segment_image_map(image_map, uf):
    segment_map = {}
    for u in uf.parent:
        segment_map[u] = image_map[uf.find(u)]
    return segment_map


def flood_fill(box, u, wall, neighbors, seen):
    boundary = [u]
    seen.add(u)
    while boundary:
        next_boundary = []
        for v in boundary:
            for d in neighbors:
                w = sum_points(v, d)
                if not in_box(w, box) or w in wall or w in seen:
                    continue
                seen.add(w)
                next_boundary.append(w)
        boundary = next_boundary


def inverse_seen(box, seen):
    xmin, xmax, ymin, ymax = box
    inverse_seen = set()
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            if (x, y) not in seen:
                inverse_seen.add((x, y))
    return inverse_seen


def fill_islands(uf, image_map, edges, neighbors=[(0, 1), (1, 0), (-1, 0), (0, -1)]):
    root = defaultdict(set)
    for u in image_map:
        root[uf.find(u)].add(u)

    for r in root:
        xmin = min(x for x, _ in root[r]) - 1
        xmax = max(x for x, _ in root[r]) + 2
        ymin = min(y for _, y in root[r]) - 1
        ymax = max(y for _, y in root[r]) + 2
        box = (xmin, xmax, ymin, ymax)
        seen = set()
        flood_fill(box, (xmin, ymin), root[r], neighbors, seen)
        unseen = inverse_seen(box, seen)
        for u in unseen:
            if not uf.union(u, r):
                continue
            edges.append((u, r))
