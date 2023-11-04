import copy

import numpy as np

from _ML import metrics

if __name__ == '__main__':

    a = [0. for i in range(25000)]
    b = copy.deepcopy(a)

    # c = 860-13-10-12
    # mmm = 1508+49+30+53

    c = 704 - 13 - 10 - 12 - 6 - 30
    mmm = 1839 + 49 + 30 + 53 + 29 + 60

    for i in range(25000):
        if i < 1335:
            a[i] = 1.0
        if c < i < c + mmm:
            b[i] = 1.0

    a = np.array(a)
    b = np.array(b)

    print(metrics.Metrics.f1_metrics(a, b))
