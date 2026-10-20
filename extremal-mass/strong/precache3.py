import sys
from concurrent.futures import ProcessPoolExecutor
def work(a):
    from mp_series import cached
    S = cached(a); return a, float(S.gamma)
if __name__ == '__main__':
    L = [float(v) for v in sys.argv[1].split(',')]
    with ProcessPoolExecutor(len(L)) as ex:
        for a, g in ex.map(work, L):
            print('cached', a, g, flush=True)
