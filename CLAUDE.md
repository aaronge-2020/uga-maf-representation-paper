# Working context for Claude sessions on this repository

Read this first in any new session. It records who is who, where the state of the project lives,
and the operational pitfalls that cost hours when forgotten. Updated 5 October 2026.

## People and roles

- **Vijay Vankadaru** (this machine's user; vj@dasion.ai / vankadaruvijay@berkeley.edu). TensorLab
  fellow. Owns the MuAt-compatible comparator, the frozen models, the HRD shortcut ablation, the
  survival baseline, feature selection, SHAP, and the results documentation.
- **Aaron Ge** (aaronge2020@gmail.com, aaronge13@icloud.com). MD candidate, UMSOM, first author.
  Owns the manuscript, the Bio MAF v4 feature matrix, the knowledge-ablation and batch-center
  analyses, Zenodo. Commits as "Aaron Ge".
- **Cindy (Xingyan) Liu** (baymincindy200@...). External validation on POG570 and PCAWG using the
  frozen models. Commits as "Xingyan-Liu".
- **Jeya Balasubramanian**. NCI staff scientist. Requested the survival baseline and the
  individual-feature-selection protocol.

## Where the state of the project lives

| Question | File |
|---|---|
| Every result, every experiment run, decisions pending | `docs/results_summary_2026-10.md` (supersedes `_2026-08.md`) |
| MuAt comparator vs the official MuAt trainer, line by line, with sources | `docs/muat_fidelity_audit_2026-10.md` |
| What changed in the code and why, July onward | `docs/CHANGELOG_vijay.md` |
| Frozen models, how to run inference, canonical OOF hash | `results/frozen_models/README.md` |
| Manuscript text as currently drafted | `submission_manuscript.docx`, `results/manuscript/text/*.md` |

If a number is needed, it is in a CSV under `results/tables/`; the summary doc names the file.
Do not quote numbers from memory; read the CSV.

## Branch and remote

- Working branch: `codex/coxnet-muat-manuscript-updates`. Remote: `aaronge-2020/uga-maf-representation-paper`.
- Large files go through Git LFS (`.gitattributes`). After cloning: `git lfs pull`.
- Windows: `git config core.longpaths true` before the first checkout (longest path is 249 chars).

## Data bundle

- Hugging Face dataset `vjdeara/cgr_tensorlab_umsom`. Fetch with `python scripts/fetch_datasets.py`.
- On Vijay's desktop the bundle is at `C:\TensorLabMarylandProject\hf_bundle\github_exports_datasets`.
  Every run takes `--datasets-dir <that path>`.
- `python scripts/audit_bundle_inputs.py` lists any missing inputs in five seconds. Run it first.
- Three files are still absent from the HF bundle (two COSMIC signature files, the Kucab raw
  directory); they are present locally. The HF copy of the OOF prediction table is one rebuild
  stale; use the repository copy (LFS, SHA-256 `84d7472c...`).

## Environment

- MuAt comparator needs CUDA. Plain `pip install -r requirements.txt` gives CPU-only torch. Then:
  `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall`
  and verify with `python -c "import torch; print(torch.cuda.is_available())"`.
- Desktop: RTX 3080 16 GB, 32 GB RAM. One MuAt tumour-typing fold at batch size 1 is 5 to 7 h.
- Console is cp1252 on Windows: scripts must print ASCII only.
- PowerShell, not cmd. `$env:VAR = 'x'` in PowerShell; `set VAR=x` in cmd.

## Rules Claude follows in this repository

1. **Never run git commands from the sandbox.** Two sessions wedged the repository with a stale
   `.git/index.lock` (long-path checkout aborts). Read `.git/refs`, `.git/logs/HEAD`, or use the
   pure-Python `dulwich` library for read-only history. Vijay stages and pushes in GitHub Desktop.
2. **No em dashes or en dashes** in any document Vijay will send. ASCII only. Check with a grep
   before presenting.
3. **Fidelity to MuAt is checked against the official source, not assumed.** The official trainer
   is `github.com/primasanjaya/muat` (`muat/trainer.py`, `muat/core.py`,
   `muat/pkg_reproduce/experiments.json`). Gradient clipping is NOT in the official recipe. The
   audit doc has the setting-by-setting table; extend it rather than restating it.
4. **Scientific decisions are the group's.** Claude fixes defects, builds analyses, documents
   deviations, and states a recommendation. Which protocol the paper reports, how survival is
   framed, whether pooling is attributed to MuAt: those go in the "Decisions needed" section, not
   into the manuscript.
5. **Verify before asserting.** Several of Aaron's October claims (6.5 epochs, 0.54 / 0.78) did not
   match the result files; several of Claude's did not either (0.539 "pre-fix", gradient clipping
   as a fidelity fix). Check the CSV, check the source, then write.
6. **Fingerprints guard checkpoints.** Any new setting that changes training must enter the
   fingerprint payload in `run_muat_style_tcga_comparator.py`, but only when non-default, so
   existing checkpoints keep their hash. Two protocols for one endpoint need distinct
   `checkpoint_subdir` values or they overwrite each other.
7. **Reruns archive, never delete.** `force_retrain_folds` renames superseded checkpoints to
   `*.superseded_<utc>`; failed attempts keep their output tag (`*_fold3_rerun_*` is the failed v1).

## Open items as of 5 October 2026

- Aaron to decide: faithful vs search protocol in the main text (recommendation: faithful in
  text, search in supplement); whether HRD moves to the faithful protocol (run is done, clean).
- Refresh the HF copy of the OOF table (needs a write token from Vijay).
- Upload the three missing bundle files to HF.
- Metastatic status as a survival confounder: unresolved since 1 August.
- Fold pairing across representations: state the limitation or rerun paired.

## Quick reproduction

```powershell
cd C:\TensorLabMarylandProject\uga-maf-representation-paper
python scripts/audit_bundle_inputs.py
python scripts/test_muat_tokens.py
python scripts/test_muat_divergence_guard.py
python src/run_all_experiments.py --config config/experiment_settings.<name>.yaml --paths config/paths.yaml --datasets-dir ..\hf_bundle\github_exports_datasets
```
