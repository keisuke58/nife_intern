# Agent Instructions — nife

Oral biofilm community dynamics (NIFE / SIIRI TRR-298). Research codebase — scripts produce fits, JSON, and figures.

## Start here

| Task | Command |
|------|---------|
| Light reproduce | `./scripts/reproduce_core.sh` |
| LIGHT figures only | `make figures` |
| Paper figure paths | `python dieckow_paper/make_figures.py` |
| ETL tests | `pip install -e ".[dev]" && pytest tests/ -q` |
| Thesis sync (slides) | `/thesis-sync` (see `.claude/commands/thesis-sync.md`) |

## Rules

1. **Never hardcode paper run paths** — use `paper_data.py`
2. **Guild order is canonical** — `GUILD_ORDER` in `guild_replicator_dieckow.py`
3. **Scripts in `scripts/`** keep the `# [nife-pathshim]` two-liner; run from repo root
4. **Don't remove `parents[2]` path anchors** in scripts
5. **No cluster/GPU in light sync** — see `docs/PROVENANCE.md` HEAVY rows

## HPC slash commands

| Command | Purpose |
|---------|---------|
| `/hpc` | PBS diffusion-fit sweep aggregation |
| `/gpu` | vancouver01 GPU dispatch |
| `/fish3d` | Full 3D FISH extraction |
| `/thesis-sync` | LIGHT figures + JA/EN decks |
| `/litwatch` | arXiv reading list update |

## Key docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — data flow
- [docs/PROVENANCE.md](docs/PROVENANCE.md) — figure regeneration map
- [CLAUDE.md](CLAUDE.md) — full developer guide
- [PAPER_OUTLINE.md](PAPER_OUTLINE.md) — manuscript structure
