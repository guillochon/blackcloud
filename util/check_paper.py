"""Manuscript checks for paper/main.tex (see .cursor/rules/).

  python util/check_paper.py          # all checks; exits nonzero on any violation

Checks: abstract length (arXiv 1920 characters) and forbidden macros; abstract not opening
with a citation; forbidden macros in the title; American spelling in the manuscript, figure scripts and README; multi-key
citations in chronological order; every cited key present in paper/refs.bib; banned words;
gendered pronouns for other authors; no overfull hbox or undefined citation in the last builds of
both typesettings (paper/submit.log and paper/preprint.log).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / 'paper' / 'main.tex'
BIB = ROOT / 'paper' / 'refs.bib'
SPELL_FILES = [TEX, ROOT / 'README.md', ROOT / 'run_model.py', ROOT / 'run_cells.py',
               *sorted((ROOT / 'blackcloud').glob('*.py'))]
ABSTRACT_LIMIT = 1920

# British form -> American form.  Word-boundary regexes, case-insensitive.
BRITISH = {
    r'\bcolour': 'color', r'\bcentre': 'center', r'\bcentred': 'centered',
    r'(?<![a-z])(milli|centi|kilo)?metres?\b': 'meter', r'\bneighbour': 'neighbor',
    r'\bfavour': 'favor', r'\bbehaviour': 'behavior', r'\bmodell(ed|ing)\b': 'model(ed|ing)',
    r'\blabell(ed|ing)\b': 'label(ed|ing)', r'\btravell(ed|ing|er)\b': 'travel(ed|ing)',
    r'\bsignall(ed|ing)\b': 'signal(ed|ing)', r'\bgrey': 'gray', r'\banalys(e|ed|ing)\b': 'analyz-',
    r'\b\w*(optimis|normalis|organis|recognis|minimis|maximis|characteris|synchronis|'
    r'parameteris|realis|emphasis|generalis|visualis|summaris|utilis|randomis|'
    r'stabilis|polaris|ionis|initialis|specialis|categoris|prioritis)'
    r'(e|ed|es|er|ers|ing|ation|ations)\b': '-ize/-ization',
    r'\banalogue': 'analog', r'\bdialogue': 'dialog',
    r'\bper cent\b': 'percent', r'\bcatalogue': 'catalog', r'\bprogramme': 'program',
    r'\bartefact': 'artifact', r'\bfibre': 'fiber', r'\bwhilst\b': 'while',
    r'\bamongst\b': 'among', r'\blearnt\b': 'learned', r'\bspectre': 'specter',
    r'\baluminium': 'aluminum', r'\btowards\b': 'toward', r'\bmanoeuv': 'maneuver',
    r'\bdefence\b': 'defense', r'\blicence\b': 'license', r'\bsulph': 'sulf',
}
BANNED = [r'\bhonest\w*', r'\btrap\w*']
PRONOUNS = r'\b(he|she|his|her|hers|him|himself|herself)\b'


def strip_comments(text):
    return re.sub(r'(?<!\\)%.*', '', text)


def abstract(tex):
    m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.S)
    return ' '.join(m.group(1).split()) if m else ''


def title(tex):
    """Body of \\title{...}, with nested braces."""
    m = re.search(r'\\title\s*\{', tex)
    if not m:
        return ''
    depth, i = 1, m.end()
    while i < len(tex) and depth:
        depth += {'{': 1, '}': -1}.get(tex[i], 0)
        i += 1
    return ' '.join(tex[m.end():i - 1].split())


def preamble_macros(tex):
    head = tex.split(r'\begin{document}')[0]
    return set(re.findall(r'\\(?:newcommand|renewcommand|def)\s*\{?\\(\w+)', head))


def bib_years(bib):
    out = {}
    for key, body in re.findall(r'@\w+\{([^,]+),(.*?)\n\}', bib, re.S):
        y = re.search(r'year\s*=\s*\{?(\d{4})', body)
        out[key.strip()] = int(y.group(1)) if y else 0
    return out


def main():
    problems = []
    tex = TEX.read_text(encoding='utf-8')
    body = strip_comments(tex)

    # abstract
    ab = abstract(body)
    print(f'abstract: {len(ab)} characters (limit {ABSTRACT_LIMIT})')
    if len(ab) > ABSTRACT_LIMIT:
        problems.append(f'abstract is {len(ab)} > {ABSTRACT_LIMIT} characters')
    forbidden = preamble_macros(tex) | {'citet', 'citep', 'cite', 'citealt', 'citeauthor', 'ref', 'eqref'}
    used = sorted({m for m in re.findall(r'\\(\w+)', ab) if m in forbidden})
    if used:
        problems.append('abstract uses forbidden macros: ' + ', '.join('\\' + u for u in used))
    if re.match(r'\s*\\cite', ab):
        problems.append('abstract opens with a citation command')

    # title: the same macro rule, since arXiv and Elsevier take it as pasted text too
    used = sorted({m for m in re.findall(r'\\(\w+)', title(body)) if m in forbidden})
    if used:
        problems.append('title uses forbidden macros: ' + ', '.join('\\' + u for u in used))


    # Elsevier highlights: 3-5 bullets, each <= 85 characters including spaces
    m = re.search(r'\\begin{highlights}(.*?)\\end{highlights}', body, re.S)
    if m:
        items = [' '.join(x.split()) for x in re.split(r'\\item', m.group(1))[1:]]
        if not 3 <= len(items) <= 5:
            problems.append(f'{len(items)} highlights (Elsevier wants 3-5)')
        for it in items:
            if len(it) > 85:
                problems.append(f'highlight over 85 characters ({len(it)}): {it[:40]}...')

    # spelling
    for f in SPELL_FILES:
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text(encoding='utf-8').splitlines(), 1):
            text = strip_comments(line) if f.suffix == '.tex' else line
            for pat, us in BRITISH.items():
                for m in re.finditer(pat, text, re.I):
                    problems.append(f'{f.relative_to(ROOT)}:{i}: British spelling "{m.group(0)}" -> {us}')

    # banned words and pronouns (manuscript prose only)
    doc = body.split(r'\begin{document}')[1]
    for i, line in enumerate(doc.splitlines(), 1):
        for pat in BANNED:
            for m in re.finditer(pat, line, re.I):
                problems.append(f'main.tex body line {i}: banned word "{m.group(0)}"')
        for m in re.finditer(PRONOUNS, line, re.I):
            problems.append(f'main.tex body line {i}: gendered pronoun "{m.group(0)}" (use they/them)')

    # citations
    years = bib_years(BIB.read_text(encoding='utf-8'))
    for keys in re.findall(r'\\cite\w*\{([^}]*)\}', body):
        ks = [k.strip() for k in keys.split(',')]
        for k in ks:
            if k not in years:
                problems.append(f'citation key {k} not in refs.bib')
        ys = [years.get(k, 0) for k in ks]
        if ys != sorted(ys):
            problems.append(f'citations out of chronological order: {keys}')

    # nothing may run into the margin: every overfull \hbox in either typesetting is a violation
    for job in ('submit', 'preprint'):
        log = TEX.with_name(job + '.log')
        if not log.exists() or log.stat().st_mtime < TEX.stat().st_mtime:
            print(f'  ! paper/{job}.log is missing or older than main.tex; run `latexmk -pdf submit preprint` '
                  'in paper/ and rerun to check margins')
            continue
        text = log.read_text(encoding='latin-1')
        for m in re.finditer(r'Overfull \\hbox \(([\d.]+)pt too wide\) (?:in paragraph|detected) at lines? ([\d-]+)', text):
            problems.append(f'{job}: main.tex lines {m.group(2)}: overfull hbox, {m.group(1)}pt into the margin')
        if re.search(r"Citation `[^']+' on page \d+ undefined", text):
            problems.append(f'{job}: undefined citations in the last build')

    for p in problems:
        print('  x', p)
    print('OK' if not problems else f'{len(problems)} problem(s)')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
