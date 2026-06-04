#!/usr/bin/env python3
"""
slide_generate_full.py  ③  修論全体からスライドデッキを一括生成する。
Claude 消費大（~30–60k tokens）。月1回程度の棚卸し向け。
出力は stdout に Beamer ソースを表示 → レビュー後に手動で既存スライドへ統合。

Usage:
    python scripts/figures/slide_generate_full.py --chapter ch5 --lang EN
    python scripts/figures/slide_generate_full.py --all --lang JA
"""
import argparse, re, sys, time
from pathlib import Path

THESIS_CHAPTERS = Path('/home/nishioka/LUHsummer26/30_Masterarbeit/chapters')
NIFE_DOCS       = Path('/home/nishioka/IKM_Hiwi/nife/docs')

CHAPTER_META = {
    'ch3': ('ch3_heine.tex',       'Chapter 3: GPU-Accelerated Bayesian Inference (5-species in-vitro)'),
    'ch4': ('ch4_dieckow.tex',     'Chapter 4: Clinical Dieckow cohort, gLV + metabolic sign prior'),
    'ch5': ('ch5_integration.tex', 'Chapter 5: Integration, spatial PDE, FISH depth analysis'),
}

SLIDE_SYSTEM = """You are a LaTeX Beamer author for a biofilm PhD thesis defense.
Given a thesis chapter, produce a concise slide section (4–8 frames).
Rules:
- \\section{...} at the start, then \\begin{frame}{Title}...\\end{frame} blocks
- Max 5 bullets per frame, each ≤ 12 words
- Preserve all numbers, p-values, citations (as \\cite{key} placeholders)
- Add \\includegraphics[width=0.8\\textwidth]{FIGURE_NAME} for every figure reference
- End with one "Summary" frame listing 3 key take-aways
- Output ONLY raw LaTeX
"""

def extract_chapter_text(ch: str, max_chars: int = 6000) -> str:
    """章テキストを読んで主要部分を抽出（長すぎる場合は先頭+結論部分）"""
    fname, _ = CHAPTER_META[ch]
    text = (THESIS_CHAPTERS / fname).read_text()
    # LaTeX コマンド/図環境を簡略化
    text = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '[FIGURE]', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', '[EQ]', text, flags=re.DOTALL)
    text = re.sub(r'%.*', '', text)
    if len(text) > max_chars:
        text = text[:max_chars] + '\n...[truncated]'
    return text

def call_claude(chapter_text: str, chapter_meta: str, lang: str) -> str:
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic"); sys.exit(1)
    client = anthropic.Anthropic()
    prompt = (f"Language: {lang}\nChapter context: {chapter_meta}\n\n"
              f"Chapter text:\n{chapter_text}\n\nGenerate slide section:")
    msg = client.messages.create(
        model='claude-sonnet-4-6',
        max_tokens=2500,
        system=SLIDE_SYSTEM,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return msg.content[0].text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chapter', choices=['ch3','ch4','ch5'], help='Single chapter')
    ap.add_argument('--all',     action='store_true',          help='All chapters')
    ap.add_argument('--lang',    default='EN', choices=['EN','JA'])
    args = ap.parse_args()

    chapters = list(CHAPTER_META.keys()) if args.all else ([args.chapter] if args.chapter else [])
    if not chapters:
        ap.print_help(); return

    print(f"% Auto-generated Beamer slides — {args.lang}")
    print(f"% Source: thesis chapters {chapters}")
    print(f"% Review carefully before merging into *_slides.tex\n")

    for ch in chapters:
        _, meta = CHAPTER_META[ch]
        print(f"\n% {'='*56}")
        print(f"% {meta}")
        print(f"% {'='*56}\n")
        text   = extract_chapter_text(ch)
        result = call_claude(text, meta, args.lang)
        print(result)
        if len(chapters) > 1:
            time.sleep(1)  # API rate limit

if __name__ == '__main__':
    main()
