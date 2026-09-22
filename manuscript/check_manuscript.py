"""Check that the manuscript quotes its sources and nothing else.

Three things, in the spirit of the rest of the project:

  1. every `\\N...` macro the text uses is defined in the generated
     `numbers.tex`, and every generated macro is used (an unused one usually
     means a sentence was cut and a claim silently lost);
  2. no literal decimal number survives in the body, so nothing in the prose
     can drift from the measurement;
  3. every `\\bibitem` is cited and every `\\cite` key resolves.

Run from this directory:  ../.venv/Scripts/python check_manuscript.py
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = HERE/'main.tex'
NUMBERS = HERE/'numbers.tex'

# Numbers that belong to the theory rather than to the measurement, and are
# therefore written out rather than read from the artifact.
ALLOWED = {
    '2,1',        # the SO(2,1) of the near-horizon isometry group
    '4.11', '4.12',   # equation numbers of arXiv:1010.0860v1
    '4.2',            # its section number
    '1.102101', '1.104',  # the ergosurface discrepancy, quoted verbatim
}


def main():
    text = MAIN.read_text(encoding='utf-8')
    numbers = NUMBERS.read_text(encoding='utf-8')
    problems = []

    defined = set(re.findall(r'\\newcommand\{\\(N\w+)\}', numbers))
    used = set(re.findall(r'\\(N[A-Za-z]+)', text))
    for name in sorted(used - defined):
        problems.append(f'macro \\{name} used but not generated')
    for name in sorted(defined - used):
        problems.append(f'macro \\{name} generated but never used')

    body = text.split(r'\begin{thebibliography}')[0]
    body = re.sub(r'(?m)^\s*%.*$', '', body)
    for literal in sorted(set(re.findall(r'(?<![\w.\\])\d+\.\d+', body))):
        if literal not in ALLOWED:
            problems.append(f'literal decimal {literal} in the body')

    keys = set(re.findall(r'\\bibitem\{([^}]+)\}', text))
    cited = set()
    for group in re.findall(r'\\cite\{([^}]+)\}', text):
        cited.update(part.strip() for part in group.split(','))
    for key in sorted(cited - keys):
        problems.append(f'citation {key} has no bibitem')
    for key in sorted(keys - cited):
        problems.append(f'bibitem {key} is never cited')

    if problems:
        for line in problems:
            print('FAIL ', line)
        return 1
    print(f'ok: {len(used)} generated quantities, {len(keys)} references, '
          'no literal numbers in the body')
    return 0


if __name__ == '__main__':
    sys.exit(main())
