import sys, numpy as np
from concurrent.futures import ProcessPoolExecutor
RES = [0.757001, 4.830610, 13.967932, 31.657915, 62.114554, 110.381663, 182.340493]
def ladder():
    out = []; a = 4.28415
    while a < 1100:
        if all(abs(a/r-1) > 0.03 for r in RES):
            out.append(round(a, 6))
        a *= 1.2
    return out
def work(a):
    from mp_series import cached
    import mpmath as mp
    S = cached(a)
    return a, float(S.gamma), float(S.gamma_check)
if __name__ == '__main__':
    L = ladder(); print(len(L), L, flush=True)
    with ProcessPoolExecutor(8) as ex:
        for a, g, c in ex.map(work, L):
            print(f'cached alpha={a} gamma={g:.6f} check={c:.1e}', flush=True)
