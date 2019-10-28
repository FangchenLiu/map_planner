import numpy as np
import math
import cv2

def transform(p):
    p = p/4 * 8
    return (p[0] + 4)/24, (p[1] + 4)/24

def vis_map(filename=None, size=512):
    landmark = np.load(filename)
    random_perturb = landmark + 0.3*np.random.randn(*landmark.shape)
    landmark[:-1] = random_perturb[:-1]
    np.random.shuffle(landmark)
    map = np.ones((size, size, 3), dtype=np.uint8) * 180
    x0, y0 = transform(np.array([-2, 2]))
    x1, y1 = transform(np.array([6, 6]))
    g1, g2 = transform(np.array([0, 8]))

    x0, y0 = int(x0 * size), int(y0 * size)
    x1, y1 = int(x1 * size), int(y1 * size)
    g1, g2 = int(g1 * size), int(g2 * size)

    cv2.rectangle(map, (x0, y0), (x1, y1), (0, 80, 80), 1)

    for l in range(landmark.shape[0]):
        if l % 2 == 0:
            l1, l2 = transform(landmark[l])
            if l1 > x1/size:
                l1 = x1/size - x1/size * np.random.uniform(0.1, 0.5)
            l1, l2 = int(l1 * size), int(l2 * size)
            cv2.circle(map, (l1, l2), 4, (60, 60, 86), -1)
    cv2.circle(map, (g1, g2), 11, (60, 80, 256), -1)
    ps = np.array([
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
    ], dtype=np.int32)
    cv2.fillConvexPoly(map, ps, (50, 50, 50))
    cv2.imshow('image', map)
    cv2.waitKey(0)
    cv2.imwrite(filename[-9:-4]+'.png',map)

if __name__ == '__main__':
    vis_map('../landmark/goal_set2000.npy')
