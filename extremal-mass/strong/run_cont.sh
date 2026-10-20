#!/bin/bash
# usage: run_cont.sh alpha_min ; launches bvp_cont.py on the ladder above alpha_min, pid in cont.pid
cd "$(dirname "$0")"
L=$(../../.venv/bin/python -B -c "
import precache2; l=sorted([a for a in precache2.ladder()+[4.62,4.73,4.93,5.04] if a>$1]); print(','.join(str(a) for a in l))")
nohup ../../.venv/bin/python -B -u bvp_cont.py $L >> bvp_cont_beta.log 2>&1 &
echo $! > cont.pid
