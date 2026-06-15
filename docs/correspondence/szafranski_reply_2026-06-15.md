# Reply to Szafranski — review of the Dieckow LOO-CV deck (draft 2026-06-15)

**Status:** LOCAL ONLY — do not commit/push (email policy). Send from Gmail with the
two decks attached: `docs/dieckow_slides_EN.pdf`, `docs/agora_slides_EN.pdf`.

**Subject:** Re: Progress update — guild-level replicator dynamics (Dieckow LOO-CV)

---

Dear Szymon,

Thank you very much for the careful review and for the two papers — van den Berg et al. 2022 and Oña et al. 2025. They are exactly the right framing; I have already worked them into how I position the method (ecological–metabolic *complementarity*; inference of interactions as one of several combinable approaches), and I have added both to my reference list.

I have revised the deck along your comments. Two decks are attached:

1. **Dieckow deck (revised)** — the main one.
2. **AGORA deck** — this is the supplement that answers your AGORA questions directly (it already has dedicated slides for it), so it is easier than squeezing everything into the main deck.

Point by point:

- **Citation.** You were right — PRJNA725874 is **Duran-Pinedo et al. 2021, *BMC Biology* 19:240** (not "Botelho"). I confirmed it against the NCBI BioProject record and corrected it throughout.
- **What AGORA output looks like + how P_ij = sgn(F_ij) is derived.** See the AGORA deck, slides "What AGORA2 supplies" and "Cross-feeding score → sign prior" (with a worked pFBA example, e.g. Actinobacteria secretes acetate/formate/succinate that Bacilli consume → F>0 → P=+1). I also added a short standalone example slide to the Dieckow deck.
- **TMCMC (10⁴ particles).** Expanded into plain language: transitional MCMC advances 10⁴ parameter "particles" through tempered stages from prior to posterior; the final particle cloud *is* the posterior over (A, b).
- **Slide 11 / Slide 14.** Simplified both — Slide 11 now has a plain left/right reading guide, and Slide 14 (the critical validation) is restructured into Question → Test → Result → Read. The LOO-stability panel now has an explicit "how to read".
- **Cross-cohort claim.** I agree with your concern about the timescale/clinical-state mismatch and the Actinobacteria axis. I have toned this down to a limited, strong-pair sign check, and I also ran a small verification battery to be honest about it: the two cohorts differ significantly in trajectory directionality (Dieckow = directional early colonization vs Duran-Pinedo = slow fluctuating progression), the inferred signs are reproducible across random seeds (so the agreement is not a fitting artefact), and a stratification by interaction type turned out underpowered — so I make no mechanistic "cross-feeding transfers" claim across cohorts. I am happy to walk you through this.

I would be glad to discuss the conclusions and the two review papers by phone next week — please let me know what time suits you while you are in Portugal.

Best regards,
Keisuke Nishioka
NIFE research intern / IKM HiWi, Leibniz University Hannover
