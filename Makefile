# Makefile — thesis figure + deck helpers (LOCAL, LIGHT only).
#
# SHARED LAB SERVER: nothing here submits cluster/PBS jobs, launches GPU compute,
# or runs heavy local compute. Targets are pure matplotlib (figures) and
# pandoc->xelatex (decks). Run from the repo root.
#
# EXCLUDED on purpose (heavy / need extra stacks — see docs/PROVENANCE.md):
#   - scripts/analysis/fish_3d_batch.py   (re-decodes ~84 .lif FOVs; use /fish3d)
#   - scripts/figures/generate_fig127.py  (fig2 AGORA — needs FBA/COMETS stack)
#   - scripts/figures/generate_fig3456.py (fig3 AGORA — needs FBA/COMETS stack)

PYTHON ?= python3

.DEFAULT_GOAL := help
.PHONY: help figures decks reproduce

help:  ## Show this help
	@echo "nife — thesis sync helpers (LOCAL, LIGHT only)"
	@echo
	@echo "Targets:"
	@echo "  make figures    Regenerate the LIGHT matplotlib data figures (no cluster/GPU)"
	@echo "  make reproduce  LIGHT figures + Dieckow paper figs + ETL tests"
	@echo "  make decks      Rebuild + upload all JA+EN decks (runs the /thesis-sync deck build)"
	@echo "  make help       Show this help (default)"
	@echo
	@echo "Excluded (heavy / extra stacks): fish_3d_batch, AGORA/FBA figs. See docs/PROVENANCE.md"

reproduce:  ## LIGHT reproduce script (figures + paper figs + pytest)
	./scripts/reproduce_core.sh

figures:  ## Regenerate the LIGHT data figures (matplotlib; NO cluster, NO GPU)
	$(PYTHON) scripts/figures/make_pipeline_overview.py
	$(PYTHON) scripts/figures/make_concept_diagram.py
	$(PYTHON) scripts/analysis/plot_depth_profiles.py
	$(PYTHON) scripts/analysis/analyze_depth_niche.py
	$(PYTHON) scripts/analysis/spatial_crossfeeding.py
	@echo "NOTE: fish_3d_batch + AGORA/FBA figs excluded (heavy / extra stacks). See docs/PROVENANCE.md"

decks:  ## Rebuild + upload all JA+EN decks via the /thesis-sync deck build (pandoc -> xelatex)
	@echo "Run the deck build from the /thesis-sync slash command"
	@echo "(.claude/commands/thesis-sync.md, step (b)): pandoc -> xelatex x2, JA + EN,"
	@echo "for: overview agora dieckow network spatial_pde defense."
	@echo "It is a Claude-driven step (texlive PATH/TEXMFHOME + per-deck pandoc invocation),"
	@echo "not a plain shell loop — invoke '/thesis-sync' (optionally with a deck subset)."
