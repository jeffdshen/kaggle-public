import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from PIL import Image

# From https://www.kaggle.com/code/ryanholbrook/getting-started-with-santa-2022
def _rotate_link(vector, direction):
    x, y = vector
    if direction == 1:  # counter-clockwise
        if y >= x and y > -x:
            x -= 1
        elif y > x and y <= -x:
            y -= 1
        elif y <= x and y < -x:
            x += 1
        else:
            y += 1
    elif direction == -1:  # clockwise
        if y > x and y >= -x:
            x += 1
        elif y >= x and y < -x:
            y += 1
        elif y < x and y <= -x:
            x -= 1
        else:
            y -= 1
    return (x, y)


# From https://www.kaggle.com/code/ryanholbrook/getting-started-with-santa-2022
def _get_square(link_length):
    link = (link_length, 0)
    coords = [link]
    for _ in range(8 * link_length - 1):
        link = _rotate_link(link, direction=1)
        coords.append(link)
    return tuple(coords)


def to_image_array(image_map, box):
    xmin, xmax, ymin, ymax = box
    image_array = np.ones((xmax - xmin, ymax - ymin, 3), dtype=np.float32)
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            if (x, y) in image_map:
                image_array[x - xmin, y - ymin] = image_map[x, y]
    return image_array


def bounds_to_size(bounds):
    xmin, xmax, ymin, ymax = bounds
    return xmax - xmin, ymax - ymin


ARMS = np.array((64, 32, 16, 8, 4, 2, 1, 1))


class ArmHelper:
    def __init__(self, arms):
        self.arms = np.array(arms)
        self.num_angles = self.arms * 8
        self.angle_to_loc = [
            {i: point for i, point in enumerate(_get_square(arm))} for arm in self.arms
        ]
        self.loc_to_angle = [
            {point: i for i, point in enumerate(_get_square(arm))} for arm in self.arms
        ]

    @staticmethod
    def rotate_single(link, direction, n):
        direction = (direction + 1) % 3 - 1
        return (link + direction) % n

    def rotate(self, config, directions):
        ns = self.num_angles
        return tuple(
            self.rotate_single(l, d, n) for l, d, n in zip(config, directions, ns)
        )

    def to_locs(self, config):
        return tuple(to_l[l] for l, to_l in zip(config, self.angle_to_loc))

    def from_locs(self, locs):
        return tuple(to_angle[l] for l, to_angle in zip(locs, self.loc_to_angle))

    def to_loc(self, config):
        locs = self.to_locs(config)
        return tuple(np.array(locs).sum(axis=0))

    @staticmethod
    def locs_to_str(locs):
        return ";".join([" ".join(map(str, loc)) for loc in locs])

    def to_str(self, config):
        return self.locs_to_str(self.to_locs(config))


def get_cost(delta_color, action, color_scale=3.0):
    return np.sqrt(np.count_nonzero(action)) + color_scale * np.abs(delta_color).sum()


def show_image(image_map, bounds, edges=[], figsize=(20, 20)):
    image_array = to_image_array(image_map, bounds)
    image_array = np.transpose(np.flip(image_array, axis=1), axes=(1, 0, 2))
    fig, ax = plt.subplots(figsize=figsize)
    xmin, xmax, ymin, ymax = bounds
    lines = []
    for x, y in edges:
        lines.append([x, y])

    lc = mc.LineCollection(lines, colors="b")
    ax.add_collection(lc)
    ax.matshow(image_array, extent=(xmin - 0.5, xmax - 0.5, ymin - 0.5, ymax - 0.5))
    return ax.grid(False)


def get_image_array(path):
    img = Image.open(path)
    img = (np.array(img) / 255)[:, :, :3]
    return np.transpose(np.flip(img, axis=0), axes=(1, 0, 2))
