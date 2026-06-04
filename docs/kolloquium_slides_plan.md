# Kolloquium スライド方針

策定日：2026-06-05 / 着手予定：2026-10月末

---

## 全体方針

Kolloquium（12月）用に **統合デッキ1本（≈30枚）** を別途作る。
既存 beamer サブデッキ（agora/dieckow/network/spatial_pde）は「詳細・専門家向け」として温存し、
統合デッキはそこから抜粋・再構成する。

---

## 統合デッキの構成案（≈30枚）

| # | セクション | 枚数 | 素材源 |
|---|-----------|------|--------|
| 1 | Introduction | 2 | 新規（臨床動機 + 研究問い） |
| 2 | Ch3：Heine Bayesian ODE | 6 | slides_nishioka_heine2025.pptx から抜粋 |
| 3 | Ch4：Dieckow + AGORA prior | 8 | dieckow_slides / agora_slides / network_slides から抜粋 |
| 4 | Ch5：空間 PDE + FISH | 8 | spatial_pde_slides / fish_pipeline_slides から抜粋 |
| 5 | Conclusion | 3 | 3本柱まとめ + Outlook 1文ずつ |
| 6 | Backup | 数枚 | defense_figures.pdf から転用 |

想定時間：20–25分発表 + QA

---

## 既存デッキの位置づけ

| デッキ | 役割 | 更新方針 |
|--------|------|---------|
| `overview_slides` | 進捗報告・全体共有（傘デッキ） | 随時更新 |
| `agora_slides` | AGORA FBA 深掘り | 結果確定後に更新 |
| `dieckow_slides` | Dieckow LOO-CV 深掘り | 結果確定後に更新 |
| `network_slides` | ネットワーク解析 | 結果確定後に更新 |
| `spatial_pde_slides` | 空間 PDE + 拡散フィット | 拡散フィット確定後に更新 |
| `fish_pipeline_slides` | FISH 解析パイプライン | COMSTAT 結果確定後に更新 |
| `slides_nishioka_heine2025.pptx` | Heine 論文用（37枚ポスター型） | 温存・変更しない |

---

## thesis figure → slide figure 変換ルール

- `thesis_style.py`（usetex/lmodern 9pt）で生成した PDF はそのまま embed 可
- フォントサイズは ≥ 16pt に上げる（スライド用に再生成 or 拡大）
- 縦長レイアウトは横並びに再配置することがある
- PROVENANCE.md に来歴記録済みの図はスライドにも生成スクリプト名を caption でメモ

---

## タイムライン上の優先順位

```
〜8月：解析・figure 確定フェーズ
        → 各 beamer サブデッキを /decks スキルで随時更新
          （まだ統合デッキは作らない）

9月〜：ch3/4/5 執筆フェーズ

10月末：Junker に draft 提出 → 統合デッキ初版を着手

11月：推敲 + 統合デッキ完成

12月：Kolloquium 本番
```

**今やること：なし。8月までは解析に集中し、beamer サブデッキを素材として育てる。**

---

## 備考

- defense_qa.pdf / defense_figures.pdf / defense_derivations.pdf（`docs/` 内）は口頭対策資料として別途存在する
- 統合デッキの形式は beamer（LaTeX）または Keynote/PPTX どちらでも可。決定は10月でよい
- 発表言語：英語（審査委員 Junker / Soleimani）
