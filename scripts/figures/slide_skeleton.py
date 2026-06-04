#!/usr/bin/env python3
"""
slide_skeleton.py  ②  修論の新しい段落からスライド骨格フレームを生成する。
Claude API を使用（~1–3k tokens/実行）。承認後に手動でスライド .tex に貼り付ける。

Usage:
    python scripts/figures/slide_skeleton.py --chapter ch5 --section "depth ordering"
    python scripts/figures/slide_skeleton.py --paragraph "段落テキストを直接渡す"
    python scripts/figures/slide_skeleton.py --diff          # git diff で最新追加段落を自動検出
"""
import argparse, re, subprocess, sys, textwrap
from pathlib import Path

THESIS_CHAPTERS = Path('/home/nishioka/LUHsummer26/30_Masterarbeit/chapters')
NIFE_DOCS       = Path('/home/nishioka/IKM_Hiwi/nife/docs')

SYSTEM_PROMPT = """You are a LaTeX Beamer slide author for a biofilm mechanics PhD thesis.
Given a thesis paragraph, produce 1–2 compact Beamer frames in LaTeX.
Rules:
- Use \\begin{frame}{Title} ... \\end{frame}
- 3–5 bullet points max per frame, each ≤ 12 words
- Keep all numbers, p-values, species names exactly as in the source
- Add \\includegraphics[width=...]{FIGURE} placeholder if a figure is referenced
- Output ONLY raw LaTeX, no explanation
- Language: match the slide file language (EN or JA specified in prompt)
"""

def get_diff_paragraphs():
    """git diff で最新コミットの thesis 段落を抽出"""
    result = subprocess.run(
        ['git', '-C', str(THESIS_CHAPTERS.parent.parent),
         'diff', 'HEAD~1', 'HEAD', '--', 'chapters/'],
        capture_output=True, text=True
    )
    added = [l[1:] for l in result.stdout.splitlines() if l.startswith('+') and not l.startswith('+++')]
    text = '\n'.join(added)
    # 段落単位に分割して長いものだけ返す
    paras = re.split(r'\n{2,}', text)
    return [p.strip() for p in paras if len(p.strip()) > 200]

def call_claude(paragraph: str, lang: str = 'EN') -> str:
    """Anthropic API 経由でスライド骨格を生成"""
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic パッケージが必要です: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()
    prompt = f"Language: {lang}\n\nThesis paragraph:\n{paragraph}\n\nGenerate Beamer frame(s):"
    msg = client.messages.create(
        model='claude-sonnet-4-6',
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return msg.content[0].text

def main():
    ap = argparse.ArgumentParser(description='Generate Beamer slide skeleton from thesis paragraph')
    ap.add_argument('--chapter',   help='Chapter name (ch3/ch4/ch5)')
    ap.add_argument('--section',   help='Section keyword to search')
    ap.add_argument('--paragraph', help='Paragraph text directly')
    ap.add_argument('--diff',      action='store_true', help='Use latest git diff')
    ap.add_argument('--lang',      default='EN', choices=['EN', 'JA'])
    args = ap.parse_args()

    paragraphs = []

    if args.paragraph:
        paragraphs = [args.paragraph]
    elif args.diff:
        paragraphs = get_diff_paragraphs()
        if not paragraphs:
            print("git diff に新しい段落が見つかりませんでした。")
            return
        print(f"{len(paragraphs)} 段落を検出しました。最初の1つを使用します。")
        paragraphs = [paragraphs[0]]
    elif args.chapter and args.section:
        ch_map = {'ch3': 'ch3_heine.tex', 'ch4': 'ch4_dieckow.tex', 'ch5': 'ch5_integration.tex'}
        fname = ch_map.get(args.chapter, f'{args.chapter}.tex')
        text  = (THESIS_CHAPTERS / fname).read_text()
        # セクション周辺の段落を抽出
        idx = text.lower().find(args.section.lower())
        if idx < 0:
            print(f"'{args.section}' が見つかりません。")
            return
        paragraphs = [text[idx:idx+800]]
    else:
        ap.print_help()
        return

    for para in paragraphs:
        print('\n' + '─' * 60)
        print('入力段落 (先頭150文字):')
        print(textwrap.shorten(para, width=150))
        print('\n▶ Beamer フレーム (Claude 生成):')
        print('─' * 60)
        result = call_claude(para, args.lang)
        print(result)
        print('─' * 60)
        print('→ 上記を対応する *_slides.tex にコピーして編集してください。')

if __name__ == '__main__':
    main()
