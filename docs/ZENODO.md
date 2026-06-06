# Zenodo Archive

Software archive for citation and long-term preservation.

## Status

Zenodo integration is **not yet enabled** for this repository. Metadata is prepared in `.zenodo.json` and `CITATION.cff`.

## Enable Zenodo

1. Sign in at https://zenodo.org with your GitHub account
2. Open https://zenodo.org/account/settings/github/
3. Toggle **ON** for `keisuke58/nife_intern`
4. Create a GitHub release (e.g. `v0.2.0`):
   ```bash
   gh release create v0.2.0 --title "v0.2.0" --notes "Reproducibility + CI release"
   ```
5. After a few minutes, check https://zenodo.org/deposit for a draft
6. Review metadata and click **Publish**
7. Add the assigned DOI to `CITATION.cff` (`doi` field + `identifiers` list)

## Cite (interim — GitHub)

Until a Zenodo DOI is assigned:

```bibtex
@software{nishioka2026nife,
  author    = {Nishioka, Keisuke},
  title     = {nife},
  year      = {2026},
  url       = {https://github.com/keisuke58/nife_intern},
  version   = {0.2.0}
}
```

For the Dieckow / peri-implantitis manuscript, cite the primary paper and this software archive once published.