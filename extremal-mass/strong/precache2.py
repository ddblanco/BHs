import numpy as np
from concurrent.futures import ProcessPoolExecutor
RES = [0.757001, 4.830610, 13.967932, 31.657915, 62.114554, 110.381663, 182.340493, 284.711377, 425.054236, 611.768766, 854.094522]
def ladder():
    out = []; a = 2.6201047
    while a < 1100:
        a *= 1.07
        if all(abs(a/r-1) > 0.02 for r in RES):
            out.append(round(a, 6))
    return out
def work(a):
    from mp_series import cached
    S = cached(a)
    return a, float(S.gamma), float(S.gamma_check)
if __name__ == '__main__':
    L = ladder(); print(len(L), L, flush=True)
    with ProcessPoolExecutor(12) as ex:
        for a, g, c in ex.map(work, L):
            print(f'cached alpha={a} gamma={g:.6f} check={c:.1e}', flush=True)
