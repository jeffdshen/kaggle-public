import numpy as np
from collections import defaultdict
from tqdm import tqdm


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
    if np.sum(np.abs(np.subtract(u, v))) > 8:
        return 8000000
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


def get_edges(image_map, neighbors=[(0, 1), (1, 0), (-1, 0), (0, -1)]):
    edges = []
    for u in image_map:
        for d in neighbors:
            v = sum_points(u, d)
            if v not in image_map:
                continue
            if v < u:
                continue
            edges.append((u, v))
    return edges


def get_all_component_edges(
    image_map, uf, neighbors=[(0, 1), (1, 0), (-1, 0), (0, -1)]
):
    edges = []
    for u in image_map:
        for d in neighbors:
            v = sum_points(u, d)
            if v not in image_map:
                continue
            if v < u:
                continue
            if uf.find(u) != uf.find(v):
                continue
            edges.append((u, v))
    return edges


def get_weighted_edges(image_map, edges):
    weighted_edges = []
    for u, v in edges:
        a = image_map[u]
        b = image_map[v]
        weighted_edges.append((get_pixel_cost(a, b, u, v), (u, v)))

    weighted_edges.sort()
    return weighted_edges


def update_segmentation(
    uf,
    component_edges,
    boxes,
    image_map,
    edges,
    max_distance,
    min_count=32,
    max_size=64,
):
    edges = get_weighted_edges(image_map, edges)

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


def prepare_segmentation(image_map):
    uf = UnionFind(image_map)
    component_edges = []
    boxes = {(x, y): (x, x + 1, y, y + 1) for x, y in image_map}
    return uf, component_edges, boxes


def run_segmentation(
    image_map,
    max_distance,
    min_count=32,
    max_size=64,
    neighbors=[(0, 1), (1, 0), (-1, 0), (0, -1)],
):
    uf, component_edges, boxes = prepare_segmentation(image_map)
    edges = get_edges(image_map, neighbors)
    update_segmentation(
        uf,
        component_edges,
        boxes,
        image_map,
        edges,
        max_distance,
        min_count,
        max_size,
    )
    return uf, component_edges, boxes


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


## Stuff for concorde


def to_edge_list(image_map, edges, m=100):
    nodes = sorted(image_map.keys())
    to_node = {u: i for i, u in enumerate(nodes)}
    edge_list = [(int(w * m + 0.5), (to_node[u], to_node[v])) for w, (u, v) in edges]
    return len(nodes), edge_list


def write_edge_list(n, edge_list, path):
    with open(path, "w") as f:
        f.write(f"{n} {len(edge_list) * 2}\n")
        for w, (u, v) in edge_list:
            f.write(f"{u} {v} {w}\n")
            f.write(f"{v} {u} {w}\n")


def tour_to_edges(image_map, tour_list):
    nodes = sorted(image_map.keys())
    tour_edges = []
    for i, j in zip(tour_list[:-1], tour_list[1:]):
        tour_edges.append((nodes[i], nodes[j]))

    tour_edges.append((nodes[tour_list[0]], nodes[tour_list[-1]]))
    return tour_edges


def read_tour(path):
    with open(path, "r") as f:
        _ = [int(x) for x in next(f).split()]
        array = [int(x) for line in f for x in line.split()]
        return array


def initial_tour(image_map):
    neighbors = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    nodes = sorted(image_map.keys())
    to_node = {u: i for i, u in enumerate(nodes)}
    seen = set()
    tour = []
    u = nodes[0]
    while len(seen) < len(image_map):
        if u in seen:
            raise ValueError(f"Could not tour, hit {len(tour)} nodes")
        tour.append(to_node[u])
        seen.add(u)
        for i, d in enumerate(neighbors):
            v = sum_points(u, d)
            if v not in image_map:
                continue
            if v not in seen:
                u = v
                if i == len(neighbors) - 1:
                    neighbors[-1], neighbors[-2] = neighbors[-2], neighbors[-1]
                break

    return tour


def get_neighbors(n):
    return [
        (x, y)
        for x in range(-n, n + 1)
        for y in range(-n, n + 1)
        if 0 < np.abs([x, y]).sum() <= n
    ]


def write_tour(tour, path):
    with open(path, "w") as f:
        f.write(f"{len(tour)}\n")
        for u in tour:
            f.write(f"{u}\n")
