"""Build the manuscript from sections/*.md (syntax in SPEC.md).
Outputs: main.tex (pdfLaTeX, arXiv-compatible), paper.html, paper.pdf (headless Edge/Chrome print), arxiv_upload.zip.
usage: python build.py [--no-pdf]   (CPU python, standard library only)"""
import html, json, os, re, shutil, subprocess, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SEC = os.path.join(HERE, 'sections')
FIGSRC = os.path.join(os.path.dirname(HERE), 'fig')
ORDER = ['00_abstract.md', '01_introduction.md', '02_related_work.md', '03_fingerprint.md', '04_setup.md', '05_results.md',
         '06_discussion.md', '07_conclusion.md', '08_statements.md', 'A_appendix.md']
TITLE = 'A Compounding Fingerprint: Telling Abstraction from Exemplar Reuse in Context'
AUTHOR = 'Sam Ramdan'
AFFIL = 'Independent Researcher'
CONTACT = 'sr4729820@gmail.com'
ORCID = '0009-0004-7723-4240'
AI_NOTE = 'Prepared with substantial AI assistance (Claude, Anthropic); see Statements.'
ERR, WARN = [], []

# ------------------------------------------------------------------ unicode -> LaTeX (text mode) for pdfLaTeX
UNI_TEX = {
    '−': '$-$', '–': '--', '—': '---', '×': '$\\times$', '≥': '$\\geq$', '≤': '$\\leq$', '≈': '$\\approx$', '≠': '$\\neq$',
    '→': '$\\to$', '←': '$\\leftarrow$', '↔': '$\\leftrightarrow$', '≫': '$\\gg$', '≪': '$\\ll$', '±': '$\\pm$', '·': '$\\cdot$',
    '…': '\\ldots{}', '∞': '$\\infty$', '∈': '$\\in$', '⌈': '$\\lceil$', '⌉': '$\\rceil$', '⌊': '$\\lfloor$', '⌋': '$\\rfloor$',
    'ρ': '$\\rho$', 'α': '$\\alpha$', 'β': '$\\beta$', 'γ': '$\\gamma$', 'δ': '$\\delta$', 'σ': '$\\sigma$', 'θ': '$\\theta$',
    'μ': '$\\mu$', 'λ': '$\\lambda$', 'ε': '$\\varepsilon$', 'Δ': '$\\Delta$', '²': '$^{2}$', '³': '$^{3}$', '¹': '$^{1}$',
    '⁰': '$^{0}$', '₀': '$_{0}$', '₁': '$_{1}$', '₂': '$_{2}$', '₃': '$_{3}$', '₄': '$_{4}$', '₆': '$_{6}$', '₈': '$_{8}$',
    '✓': '$\\checkmark$', '†': '\\textsuperscript{\\dag}', '‡': '\\textsuperscript{\\ddag}', '§': '\\S{}', '°': '$^{\\circ}$',
    '“': '``', '”': "''", '‘': '`', '’': "'", '\u00a0': '~', '…': '\\ldots{}', 'é': "\\'e", 'è': '\\`e', 'ö': '\\"o', 'ü': '\\"u',
    'ä': '\\"a', 'ó': "\\'o", 'á': "\\'a", 'í': "\\'i", 'ñ': '\\~n', '≡': '$\\equiv$', '⊕': '$\\oplus$', '√': '$\\surd$',
    'ℤ': '$\\mathbb{Z}$', '∖': '$\\setminus$', '∣': '$\\mid$', '⋯': '$\\cdots$', '•': '$\\bullet$', '✗': '$\\times$',
}
MATH_OK = re.compile(r'\\(geq|leq|times|approx|cdot|ldots|to|in|neq|pm|infty|lfloor|rfloor|lceil|rceil|alpha|beta|gamma|delta|Delta|rho|sigma|theta|mu|mathrm|text|frac|gg|ll)\b')
GREEK_HTML = {'Delta': 'Δ', 'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'rho': 'ρ', 'sigma': 'σ', 'theta': 'θ', 'mu': 'μ'}
SYM_HTML = {'geq': '≥', 'leq': '≤', 'times': '×', 'approx': '≈', 'cdot': '·', 'ldots': '…', 'to': '→', 'in': '∈', 'neq': '≠',
            'pm': '±', 'infty': '∞', 'lfloor': '⌊', 'rfloor': '⌋', 'lceil': '⌈', 'rceil': '⌉', 'gg': '≫', 'll': '≪'}


def tex_text(s):
    """escape plain text for LaTeX (no math, no markup)."""
    out = []
    for ch in s:
        if ch in '&%$#_{}':
            out.append('\\' + ch)
        elif ch == '~':
            out.append('\\textasciitilde{}')
        elif ch == '^':
            out.append('\\textasciicircum{}')
        elif ch == '\\':
            out.append('\\textbackslash{}')
        elif ch == '<':
            out.append('$<$')
        elif ch == '>':
            out.append('$>$')
        elif ch == '|':
            out.append('$|$')
        elif ord(ch) > 127:
            if ch in UNI_TEX:
                out.append(UNI_TEX[ch])
            else:
                ERR.append(f'no LaTeX mapping for character {ch!r} (U+{ord(ch):04X})'); out.append('?')
        else:
            out.append(ch)
    return ''.join(out)


def check_math(m, where):
    for cmd in re.findall(r'\\([A-Za-z]+)', m):
        if not MATH_OK.match('\\' + cmd):
            ERR.append(f'{where}: math command \\{cmd} not allowed in ${m}$')
    for ch in m:
        if ord(ch) > 127:
            ERR.append(f'{where}: non-ASCII character {ch!r} inside math ${m}$ (use LaTeX commands)')


def math_html(m):
    s = m
    s = re.sub(r'\\(mathrm|text)\{([^{}]*)\}', lambda x: f'<span class="mr">{html.escape(x.group(2))}</span>', s)
    s = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', lambda x: f'({x.group(1)})/({x.group(2)})', s)
    s = re.sub(r'\\(' + '|'.join(GREEK_HTML) + r')\b', lambda x: GREEK_HTML[x.group(1)], s)
    s = re.sub(r'\\(' + '|'.join(SYM_HTML) + r')\b', lambda x: SYM_HTML[x.group(1)], s)
    for _ in range(3):
        s = re.sub(r'_\{([^{}]*)\}', r'<sub>\1</sub>', s)
        s = re.sub(r'\^\{([^{}]*)\}', r'<sup>\1</sup>', s)
    s = re.sub(r'_([A-Za-z0-9])', r'<sub>\1</sub>', s)
    s = re.sub(r'\^([A-Za-z0-9])', r'<sup>\1</sup>', s)
    s = s.replace('-', '−')
    # italicise single latin letters that are not inside tags
    parts = re.split(r'(<[^>]+>)', s)
    in_mr = 0
    for i, p in enumerate(parts):
        if p.startswith('<'):
            in_mr += 1 if p.startswith('<span class="mr"') else (-1 if p == '</span>' and in_mr else 0)
            continue
        if not in_mr:
            parts[i] = re.sub(r'(?<![A-Za-z])([A-Za-z])(?![A-Za-z])', r'<i>\1</i>', p)
    return '<span class="math">' + ''.join(parts) + '</span>'


# ------------------------------------------------------------------ parse
HEAD = re.compile(r'^(#{2,4}) (.+?)(?: \{#((?:sec|app):[\w-]+)\})?\s*$')
FIG = re.compile(r'^!\[(.+)\]\((fig/[\w.-]+)\)\{#(fig:[\w-]+)\}\s*$')
TABCAP = re.compile(r'^Table: (.+?) \{#(tab:[\w-]+)\}\s*$')


def parse(path):
    lines = open(path, encoding='utf-8').read().replace('\r\n', '\n').split('\n')
    blocks, i = [], 0
    while i < len(lines):
        l = lines[i]
        if not l.strip():
            i += 1; continue
        m = HEAD.match(l)
        if m:
            blocks.append(('h', len(m.group(1)), m.group(2).strip(), m.group(3))); i += 1; continue
        m = FIG.match(l)
        if m:
            blocks.append(('fig', m.group(1), m.group(2), m.group(3))); i += 1; continue
        m = TABCAP.match(l)
        if m:
            cap, lab = m.group(1), m.group(2); i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append(lines[i].strip()); i += 1
            if len(rows) < 3 or not re.match(r'^\|[\s:|-]+\|$', rows[1]):
                ERR.append(f'{os.path.basename(path)}: table {lab} malformed'); continue
            cells = [[c.strip() for c in r.strip('|').split('|')] for r in rows]
            hdr, body = cells[0], cells[2:]
            for r in body:
                if len(r) != len(hdr):
                    ERR.append(f'{os.path.basename(path)}: table {lab} row has {len(r)} cells, header {len(hdr)}: {r[:2]}')
            blocks.append(('tab', cap, lab, hdr, body)); continue
        if l.strip().startswith('|'):
            ERR.append(f'{os.path.basename(path)}: table without "Table: ... {{#tab:...}}" caption near: {l[:60]}')
            while i < len(lines) and lines[i].strip().startswith('|'):
                i += 1
            continue
        if re.match(r'^\s*(- |\d+\. )', l):
            ordered = bool(re.match(r'^\s*\d+\. ', l)); items = []
            while i < len(lines) and re.match(r'^\s*(- |\d+\. )', lines[i]):
                item = re.sub(r'^\s*(- |\d+\. )', '', lines[i]); i += 1
                while i < len(lines) and lines[i].strip() and not re.match(r'^\s*(- |\d+\. |#|\||!\[|Table: )', lines[i]):
                    item += ' ' + lines[i].strip(); i += 1
                if re.match(r'^\s{2,}(- |\d+\. )', item):
                    ERR.append(f'{os.path.basename(path)}: nested list not allowed')
                items.append(item)
            blocks.append(('list', ordered, items)); continue
        if l.lstrip().startswith('>'):
            ERR.append(f'{os.path.basename(path)}: blockquote not allowed: {l[:50]}')
        para = [l.strip()]; i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r'^(#{2,4} |!\[|Table: |\||\s*- |\s*\d+\. )', lines[i]):
            para.append(lines[i].strip()); i += 1
        blocks.append(('p', ' '.join(para)))
    return blocks


# ------------------------------------------------------------------ numbering and references
def number(all_blocks):
    labels, cite_order = {}, []
    sec = sub = 0; app = 0; in_app = False; fig = tab = 0
    for fname, blocks in all_blocks:
        if fname == 'A_appendix.md':
            in_app = True
        for b in blocks:
            if b[0] == 'h':
                lvl, title, lab = b[1], b[2], b[3]
                if lvl == 2 and title.lower() not in ('abstract', 'statements', 'references'):
                    if in_app:
                        app += 1; sub = 0; num = chr(64 + app)
                    else:
                        sec += 1; sub = 0; num = str(sec)
                    if lab:
                        labels[lab] = ('Appendix ' if in_app else 'Section ') + num
                elif lvl == 3:
                    sub += 1; num = (chr(64 + app) if in_app else str(sec)) + f'.{sub}'
                    if lab:
                        labels[lab] = ('Appendix ' if in_app else 'Section ') + num
                elif lab:
                    labels[lab] = 'Section'
            elif b[0] == 'fig':
                fig += 1; labels[b[3]] = f'Figure {fig}'
            elif b[0] == 'tab':
                tab += 1; labels[b[2]] = f'Table {tab}'
            texts = [b[1]] if b[0] == 'p' else (b[2] if b[0] == 'list' else ([b[1]] if b[0] in ('fig', 'tab') else []))
            if b[0] == 'tab':
                texts += [c for r in [b[3]] + b[4] for c in r]
            for t in texts:
                for grp in re.findall(r'\[(@[^\]]+)\]', t):
                    for k in re.findall(r'@([\w-]+)', grp):
                        if k not in cite_order:
                            cite_order.append(k)
    return labels, cite_order


def inline(s, mode, labels, cites, where):
    """mode 'tex' or 'html'. Order: protect math and code, citations, refs, then escape + emphasis."""
    store = []

    def keep(x):
        store.append(x); return f'\x00{len(store) - 1}\x00'
    # math
    def do_math(m):
        body = m.group(1); check_math(body, where)
        return keep(f'${body}$' if mode == 'tex' else math_html(body))
    s = re.sub(r'\$([^$]+)\$', do_math, s)
    # code; a code span holding only a URL becomes a link
    def do_code(m):
        c = m.group(1)
        if re.fullmatch(r'https?://[A-Za-z0-9./_~:-]+', c):
            return keep(f'\\url{{{c}}}' if mode == 'tex' else f'<a href="{html.escape(c)}"><code>{html.escape(c)}</code></a>')
        return keep('\\texttt{' + re.sub(r'-(?=-)', '-{}', tex_text(c)).replace('/', '/\\allowbreak{}') + '}' if mode == 'tex'
                    else '<code>' + html.escape(c) + '</code>')
    s = re.sub(r'`([^`]+)`', do_code, s)
    # citations
    def do_cite(m):
        keys = re.findall(r'@([\w-]+)', m.group(1))
        for k in keys:
            if k not in cites:
                ERR.append(f'{where}: unknown citation key {k}')
        if mode == 'tex':
            return keep('~\\cite{' + ','.join(keys) + '}')
        nums = sorted(cites.index(k) + 1 for k in keys if k in cites)
        return keep(' [' + ', '.join(f'<a href="#ref-{n}">{n}</a>' for n in nums) + ']')
    s = re.sub(r'\s*\[(@[^\]]+)\]', do_cite, s)
    # cross references
    def do_ref(m):
        lab = f'{m.group(1)}:{m.group(2)}'
        if lab not in labels:
            ERR.append(f'{where}: unknown label {lab}'); return keep('??')
        word, _, num = labels[lab].partition(' ')
        if mode == 'tex':
            return keep(f'{word}~\\ref{{{lab}}}')
        return keep(f'<a href="#{lab.replace(":", "-")}">{word}&nbsp;{num}</a>')
    s = re.sub(r'(?<![\w@])@(sec|fig|tab|app):([\w-]+)', do_ref, s)
    if re.search(r'<[a-zA-Z/][^>]*>', s) or re.search(r'\\[a-zA-Z]+', s):
        ERR.append(f'{where}: raw HTML/LaTeX outside math: {s[:80]}')
    # quotes -> curly
    s = re.sub(r'(^|[\s(\[\u2014\u2013/])"', '\\1\u201c', s); s = s.replace('"', '\u201d')
    s = re.sub(r"(^|[\s(\[])'", '\\1\u2018', s)
    # escape + emphasis
    if mode == 'tex':
        s = tex_text(s)
        s = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', s)
        s = re.sub(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', r'\\emph{\1}', s)
    else:
        s = html.escape(s, quote=False)
        s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
        s = re.sub(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', r'<em>\1</em>', s)
    if '*' in re.sub(r'\x00\d+\x00', '', s):
        WARN.append(f'{where}: stray asterisk in: {s[:80]}')
    return re.sub(r'\x00(\d+)\x00', lambda m: store[int(m.group(1))], s)


# ------------------------------------------------------------------ emitters
def colspec(hdr, body):
    """-> (environment, column spec). Narrow tables: plain tabular with l columns. Wide tables: tabularx at \\linewidth,
    keeping short single-token columns as l and letting text columns wrap (raggedright X)."""
    n = len(hdr)
    clean = lambda c: re.sub(r'[*`$\\{}]', '', c)
    body = [r + [''] * (n - len(r)) for r in body]   # ragged rows are reported as errors by parse()
    lens = [max(len(clean(r[j])) for r in [hdr] + body) for j in range(n)]
    if sum(lens) + 3 * n <= 95:
        return 'tabular', '@{}' + 'l' * n + '@{}'
    cols = ['l' if all(' ' not in clean(r[j]).strip() and len(clean(r[j])) <= 14 for r in [hdr] + body) else 'X' for j in range(n)]
    if 'X' not in cols:
        cols[max(range(n), key=lambda j: lens[j])] = 'X'
    # X columns share the free width in proportion to their mean cell length, but each at least as wide as its longest
    # word (estimated at 5.2pt per character of 10pt text on a 468pt line); tabularx \hsize weights must sum to the X count
    xs = [j for j in range(n) if cols[j] == 'X']
    nx = len(xs)
    free = 468.0 - sum(5.2 * lens[j] for j in range(n) if cols[j] == 'l') - 12.0 * (n - 1)
    base = free / nx
    floor = {j: (5.2 * max(len(wd) for r in [hdr] + body for wd in (clean(r[j]).split() or [''])) + 4.0) / base for j in xs}
    mean = {j: max(4.0, sum(len(clean(r[j])) for r in body) / max(1, len(body))) for j in xs}
    if sum(floor.values()) >= nx:
        w = {j: 1.0 for j in xs}
    else:
        fixed = set()
        while True:   # proportional shares, pinning any column that falls below its floor
            rest = [j for j in xs if j not in fixed]
            room = nx - sum(floor[j] for j in fixed)
            w = {j: floor[j] for j in fixed}
            w.update({j: room * mean[j] / sum(mean[k] for k in rest) for j in rest})
            low = [j for j in rest if w[j] < floor[j]]
            if not low:
                break
            fixed.update(low)
    r4 = {j: round(w[j], 4) for j in xs}
    big = max(xs, key=lambda j: r4[j])
    r4[big] = round(nx - sum(v for j, v in r4.items() if j != big), 4)
    return 'tabularx', '@{}' + ''.join(f'>{{\\raggedright\\arraybackslash\\hsize={r4[j]:.4f}\\hsize}}X' if cols[j] == 'X' else 'l'
                                       for j in range(n)) + '@{}'


def emit(all_blocks, labels, cites, refs):
    tex, htm = [], []
    in_app = False
    for fname, blocks in all_blocks:
        where0 = fname
        if fname == 'A_appendix.md' and not in_app:
            in_app = True
            tex.append('\\clearpage\n\\appendix\n'); htm.append('<div class="pagebreak"></div><div class="appendix">')
        for bi, b in enumerate(blocks):
            where = f'{where0}#{bi}'
            if b[0] == 'h':
                lvl, title, lab = b[1], b[2], b[3]
                if fname == '00_abstract.md' and lvl == 2:
                    continue
                t_tex, t_htm = inline(title, 'tex', labels, cites, where), inline(title, 'html', labels, cites, where)
                cmd = {2: 'section', 3: 'subsection', 4: 'paragraph'}[lvl]
                star = '*' if (lvl == 2 and fname == '08_statements.md') else ''
                tex.append(f'\\{cmd}{star}{{{t_tex}}}' + (f'\\label{{{lab}}}' if lab and not star else '') + '\n')
                num = labels.get(lab, '').partition(' ')[2] if lab else ''
                anchor = f' id="{lab.replace(":", "-")}"' if lab else ''
                if lvl == 4:
                    htm.append(f'<h4{anchor}>{t_htm}</h4>')
                else:
                    htm.append(f'<h{lvl}{anchor}>' + (f'<span class="secnum">{num}</span> ' if num else '') + f'{t_htm}</h{lvl}>')
            elif b[0] == 'p':
                if fname == '00_abstract.md':
                    continue
                tex.append(inline(b[1], 'tex', labels, cites, where) + '\n')
                htm.append('<p>' + inline(b[1], 'html', labels, cites, where) + '</p>')
            elif b[0] == 'list':
                env = 'enumerate' if b[1] else 'itemize'
                tex.append(f'\\begin{{{env}}}\n' + ''.join('  \\item ' + inline(it, 'tex', labels, cites, where) + '\n' for it in b[2]) + f'\\end{{{env}}}\n')
                tag = 'ol' if b[1] else 'ul'
                htm.append(f'<{tag}>' + ''.join('<li>' + inline(it, 'html', labels, cites, where) + '</li>' for it in b[2]) + f'</{tag}>')
            elif b[0] == 'fig':
                cap, path, lab = b[1], b[2], b[3]
                if not os.path.exists(os.path.join(FIGSRC, os.path.basename(path))):
                    ERR.append(f'{where}: missing figure {path}')
                pdfp = path[:-4] + '.pdf'
                tex.append('\\begin{figure}[tbp]\n\\centering\n\\includegraphics[width=\\linewidth,height=0.42\\textheight,keepaspectratio]{' + pdfp + '}\n\\caption{' +
                           inline(cap, 'tex', labels, cites, where) + '}\\label{' + lab + '}\n\\end{figure}\n')
                htm.append(f'<figure id="{lab.replace(":", "-")}"><img src="{path}" alt=""><figcaption><b>{labels[lab]}.</b> ' +
                           inline(cap, 'html', labels, cites, where) + '</figcaption></figure>')
            elif b[0] == 'tab':
                cap, lab, hdr, body = b[1], b[2], b[3], b[4]
                env, spec = colspec(hdr, body)
                rows_t = [' & '.join(inline(c, 'tex', labels, cites, where) for c in hdr) + ' \\\\\n\\midrule\n'] + \
                         [' & '.join(inline(c, 'tex', labels, cites, where) for c in r) + ' \\\\\n' for r in body]
                tex.append('\\begin{table}[tbp]\n\\centering\n\\small\n\\caption{' + inline(cap, 'tex', labels, cites, where) + '}\\label{' + lab +
                           '}\n\\begin{' + env + '}' + ('{\\linewidth}' if env == 'tabularx' else '') + '{' + spec + '}\n\\toprule\n' +
                           ''.join(rows_t) + '\\bottomrule\n\\end{' + env + '}\n\\end{table}\n')
                htm.append(f'<div class="tablewrap" id="{lab.replace(":", "-")}"><div class="tcap"><b>{labels[lab]}.</b> ' + inline(cap, 'html', labels, cites, where) +
                           '</div><table><thead><tr>' + ''.join('<th>' + inline(c, 'html', labels, cites, where) + '</th>' for c in hdr) +
                           '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + inline(c, 'html', labels, cites, where) + '</td>' for c in r) + '</tr>' for r in body) +
                           '</tbody></table></div>')
        if fname == '07_conclusion.md':
            pass
    if in_app:
        htm.append('</div>')
    return tex, htm


def ref_entry(k, r, mode):
    parts = [r['authors'].rstrip('.'), r['title'].rstrip('.'), r['venue'], r['year']]
    if mode == 'tex':
        a, t, v, y = [tex_text(x) for x in parts]
        url = r.get('url', '')
        return f'\\bibitem{{{k}}} {a}. \\newblock {t}. \\newblock {v}, {y}.' + (f' \\newblock \\url{{{url}}}' if url else '')
    a, t, v, y = [html.escape(x) for x in parts]
    url = r.get('url', '')
    return f'{a}. {t}. <i>{v}</i>, {y}.' + (f' <a href="{html.escape(url)}">{html.escape(url)}</a>' if url else '')


TEX_PRE = r"""%% Generated by build.py from sections/*.md -- edit the Markdown sources, not this file.
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx,booktabs,array,calc,tabularx}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{enumitem}
\setlist{itemsep=1pt,topsep=3pt}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\usepackage{url}
\hypersetup{pdftitle={%TITLE%},pdfauthor={%AUTHOR%}}
\setlength{\parskip}{2pt}
\renewcommand{\topfraction}{0.85}
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.6}
\setcounter{topnumber}{3}
\setcounter{totalnumber}{4}
\extrafloats{40}
\widowpenalty=10000
\clubpenalty=10000
\usepackage[section]{placeins}
\title{%TITLE%}
\author{%AUTHOR%\\ {\small %AFFIL%}\\ {\small \texttt{%CONTACT%} \quad ORCID \href{https://orcid.org/%ORCID%}{%ORCID%}}\\[4pt] {\small\itshape %AI_NOTE%}}
\date{}
\begin{document}
\maketitle
\begin{abstract}
%ABSTRACT%
\end{abstract}
"""

CSS = """
@page { size: A4; margin: 2.1cm 2.2cm 2.3cm 2.2cm; }
body { font-family: 'Times New Roman', Times, 'Liberation Serif', serif; font-size: 10.8pt; line-height: 1.38; color: #111; max-width: 46em; margin: 0 auto; }
h1.title { font-size: 17pt; text-align: center; margin: 0.2em 0 0.3em; line-height: 1.25; }
.author { text-align: center; margin-bottom: 1.2em; }
.abstract { margin: 0 2.2em 1.4em; font-size: 10pt; }
.abstract h2 { font-size: 10.5pt; text-align: center; margin: 0 0 0.3em; }
h2 { font-size: 13pt; margin: 1.3em 0 0.4em; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 1.0em 0 0.3em; page-break-after: avoid; }
h4 { font-size: 10.8pt; display: inline; margin: 0; padding-right: 0.5em; }
h4 + p { display: inline; }
h4 + p::after { content: ''; display: block; margin-bottom: 0.55em; }
p { margin: 0 0 0.55em; text-align: justify; hyphens: auto; }
.secnum { margin-right: 0.35em; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 0.88em; }
.math i { font-style: italic; } .math .mr { font-style: normal; }
figure { margin: 1em 0 1.2em; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 9.5cm; }
figcaption { font-size: 9.6pt; text-align: justify; margin-top: 0.35em; }
.tablewrap { margin: 0.9em 0 1.2em; page-break-inside: avoid; font-size: 9.4pt; }
.tcap { margin-bottom: 0.3em; text-align: justify; font-size: 9.6pt; }
table { border-collapse: collapse; width: 100%; border-top: 1.4px solid #000; border-bottom: 1.4px solid #000; }
th { border-bottom: 0.8px solid #000; text-align: left; padding: 3px 5px; font-weight: bold; vertical-align: bottom; }
td { padding: 2.5px 5px; vertical-align: top; text-align: left; }
ul, ol { margin: 0.2em 0 0.6em 1.4em; padding: 0; } li { margin-bottom: 0.2em; text-align: justify; }
.refs { font-size: 9.4pt; } .refs li { margin-bottom: 0.25em; }
.pagebreak { page-break-before: always; }
a { color: #1a3f8f; text-decoration: none; }
"""


def main():
    nopdf = '--no-pdf' in sys.argv
    refs = json.load(open(os.path.join(HERE, 'refs.json'), encoding='utf-8'))
    all_blocks = [(f, parse(os.path.join(SEC, f))) for f in ORDER if os.path.exists(os.path.join(SEC, f))]
    missing = [f for f in ORDER if not os.path.exists(os.path.join(SEC, f))]
    if missing:
        WARN.append(f'missing section files: {missing}')
    labels, cite_order = number(all_blocks)
    for k in cite_order:
        if k not in refs:
            ERR.append(f'citation key not in refs.json: {k}')
    cites = [k for k in cite_order if k in refs]
    # duplicate labels
    seen = {}
    for f, bl in all_blocks:
        for b in bl:
            lab = b[3] if b[0] in ('h', 'fig') else (b[2] if b[0] == 'tab' else None)
            if lab:
                if lab in seen:
                    ERR.append(f'duplicate label {lab} ({seen[lab]} and {f})')
                seen[lab] = f
    abstract = [b[1] for f, bl in all_blocks if f == '00_abstract.md' for b in bl if b[0] == 'p']
    tex_body, htm_body = emit(all_blocks, labels, cites, refs)
    # --- LaTeX
    tex = TEX_PRE.replace('%TITLE%', tex_text(TITLE)).replace('%AUTHOR%', tex_text(AUTHOR)).replace('%AFFIL%', tex_text(AFFIL)).replace('%CONTACT%', tex_text(CONTACT)).replace('%ORCID%', ORCID).replace('%AI_NOTE%', tex_text(AI_NOTE)).replace(
        '%ABSTRACT%', '\n\n'.join(inline(a, 'tex', labels, cites, 'abstract') for a in abstract))
    body = '\n'.join(tex_body)
    app_at = body.find('\\clearpage\n\\appendix')
    bib = '\\begin{thebibliography}{99}\n\\small\n' + '\n'.join(ref_entry(k, refs[k], 'tex') for k in cites) + '\n\\end{thebibliography}\n'
    body = (body[:app_at] + bib + body[app_at:]) if app_at >= 0 else body + bib
    tex += body + '\n\\end{document}\n'
    open(os.path.join(HERE, 'main.tex'), 'w', encoding='utf-8').write(tex)
    # figures
    figdir = os.path.join(HERE, 'fig'); os.makedirs(figdir, exist_ok=True)
    for f in os.listdir(FIGSRC):
        if f.endswith(('.png', '.pdf')):
            shutil.copy2(os.path.join(FIGSRC, f), os.path.join(figdir, f))
    # --- HTML
    refs_html = '<h2 id="sec-references">References</h2><ol class="refs">' + ''.join(
        f'<li id="ref-{i + 1}">{ref_entry(k, refs[k], "html")}</li>' for i, k in enumerate(cites)) + '</ol>'
    body_h = '\n'.join(htm_body)
    app_h = body_h.find('<div class="pagebreak"></div><div class="appendix">')
    body_h = (body_h[:app_h] + refs_html + body_h[app_h:]) if app_h >= 0 else body_h + refs_html
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(TITLE)}</title><style>{CSS}</style></head><body>'
            f'<h1 class="title">{html.escape(TITLE)}</h1><div class="author">{html.escape(AUTHOR)}<br><small>{html.escape(AFFIL)}<br>{html.escape(CONTACT)} &nbsp; ORCID <a href="https://orcid.org/{ORCID}">{ORCID}</a><br><i>{html.escape(AI_NOTE)}</i></small></div>'
            '<div class="abstract"><h2>Abstract</h2>' + ''.join('<p>' + inline(a, 'html', labels, cites, 'abstract') + '</p>' for a in abstract) +
            '</div>' + body_h + '</body></html>')
    open(os.path.join(HERE, 'paper.html'), 'w', encoding='utf-8').write(page)
    # --- arXiv bundle
    # fixed entry dates, LF line endings and 'made by' field, so that the bundle is byte-identical on every OS
    def zadd(z, src, arc):
        zi = zipfile.ZipInfo(arc, date_time=(2026, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.create_system = 3
        data = open(src, 'rb').read()
        z.writestr(zi, data.replace(b'\r\n', b'\n') if arc.endswith('.tex') else data)
    with zipfile.ZipFile(os.path.join(HERE, 'arxiv_upload.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
        zadd(z, os.path.join(HERE, 'main.tex'), 'main.tex')
        for f in sorted(os.listdir(figdir)):
            if f.endswith('.pdf'):
                zadd(z, os.path.join(figdir, f), f'fig/{f}')
    # --- report
    words = {f: sum(len(re.sub(r'[*`$]', '', (b[1] if b[0] == 'p' else ' '.join(b[2]) if b[0] == 'list' else '')).split()) for b in bl) for f, bl in all_blocks}
    print('words per file:', words, '| main text total:', sum(v for k, v in words.items() if k[0] in '01234567'))
    print('labels:', len(labels), '| citations:', len(cites), '| figures:', sum(1 for v in labels.values() if v.startswith('Figure')),
          '| tables:', sum(1 for v in labels.values() if v.startswith('Table')))
    for w in WARN:
        print('WARN', w)
    for e in ERR:
        print('ERROR', e)
    if ERR:
        print(f'{len(ERR)} errors'); sys.exit(1)
    if not nopdf:
        if not latex_pdf():
            print_pdf()


def latex_pdf():
    """compile the arXiv bundle with pdfLaTeX (MiKTeX or TeX Live) into paper.pdf; True on success. Fixed SOURCE_DATE_EPOCH
    makes the PDF byte-reproducible for a given TeX installation. Falls back to the HTML print when no engine is found."""
    import tempfile
    cands = [shutil.which('pdflatex'), os.path.expandvars(r'%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe'),
             r'C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe']
    exe = next((c for c in cands if c and os.path.exists(c)), None)
    if not exe:
        print('PDF: no pdflatex found; printing paper.html instead')
        return False
    out = os.path.join(HERE, 'paper.pdf')
    env = dict(os.environ, SOURCE_DATE_EPOCH='1767225600', FORCE_SOURCE_DATE='1')   # 2026-01-01T00:00:00Z
    with tempfile.TemporaryDirectory(prefix='texbuild_', ignore_cleanup_errors=True) as d:
        with zipfile.ZipFile(os.path.join(HERE, 'arxiv_upload.zip')) as z:
            z.extractall(d)
        for _ in range(3):
            r = subprocess.run([exe, '-interaction=nonstopmode', '-halt-on-error', 'main.tex'], cwd=d, env=env,
                               capture_output=True, timeout=600)
            if r.returncode != 0:
                print('PDF: pdflatex failed; see main.log:\n' + open(os.path.join(d, 'main.log'), encoding='latin-1').read()[-2000:])
                sys.exit(2)
        log = open(os.path.join(d, 'main.log'), encoding='latin-1').read()
        issues = {k: log.count(k) for k in ('LaTeX Warning', 'Overfull', 'undefined')}
        os.replace(os.path.join(d, 'main.pdf'), out)
    print('PDF:', out, '(pdfLaTeX;', ', '.join(f'{k}: {v}' for k, v in issues.items()) + ')')
    return True


def print_pdf():
    """print paper.html to paper.pdf with headless Chrome/Chromium/Edge; the committed PDF is replaced only on success."""
    import pathlib, tempfile
    cands = [r'C:\Program Files\Google\Chrome\Application\chrome.exe', r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
             '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
             r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe', r'C:\Program Files\Microsoft\Edge\Application\msedge.exe']
    cands += [shutil.which(n) for n in ('google-chrome', 'chromium', 'chromium-browser', 'chrome', 'msedge', 'microsoft-edge')]
    exes = [e for i, e in enumerate(cands) if e and os.path.exists(e) and e not in cands[:i]]
    if not exes:
        print('PDF: skipped (no Chrome, Chromium or Edge found); paper.pdf left unchanged')
        return
    out = os.path.join(HERE, 'paper.pdf')
    tmp = out + '.tmp.pdf'
    url = pathlib.Path(HERE, 'paper.html').as_uri()
    for exe in exes:
        if os.path.exists(tmp):
            os.remove(tmp)
        with tempfile.TemporaryDirectory(prefix='msbuild_', ignore_cleanup_errors=True) as prof:
            subprocess.run([exe, '--headless=new', '--disable-gpu', '--no-pdf-header-footer', '--run-all-compositor-stages-before-draw',
                            f'--user-data-dir={prof}', '--no-first-run',
                            '--virtual-time-budget=5000', f'--print-to-pdf={tmp}', url], capture_output=True, timeout=180)
        if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, out)
            print('PDF:', out)
            return
        print('PDF: failed with', exe)
    print('PDF: every browser failed; paper.pdf left unchanged')
    sys.exit(2)


if __name__ == '__main__':
    main()
