# Using this project directly with GitHub

Use the existing `stateful-code-completion` project directory as the repository
root. There is no separate release folder, generated copy or ZIP. Keep this
README, `src/`, `tests/`, `notebooks/`, `reports/` and the project `.gitignore`
together at that root.

The enclosing Growth workspace intentionally ignores lab directories. Create or
use a repository rooted at this project itself rather than force-adding its
contents to the parent repository. This preparation does not initialize a
repository, create a remote, commit, push or publish anything.

## What Git includes

| Included | Purpose |
|---|---|
| `src/`, `tests/`, `scripts/` | Original implementation, correctness tests and validation-audit helper |
| `notebooks/stateful_code_completion.ipynb` | Original executed notebook, with code, outputs and execution counts unchanged |
| `reports/run_001/` | Curated report, figures, tables, aggregate evidence, checksums and provenance |
| `README.md`, `docs/sequence-systems.md`, this guide | Setup/run instructions and connected experimental evidence |
| `requirements.in`, `requirements.txt`, `.python-version` | Dependency specifications, hashed lock and Python version |
| `LICENSE`, `THIRD_PARTY_NOTICES.md`, test/lint settings | License boundaries and development configuration |

Git ignores environments, dataset/tokenizer directories, full `runs/`, weights,
arrays, measurement logs, tuning files, caches, credentials, private drafts and
historical local receipts. Ignoring a file leaves it on disk. The original run
and checkpoint files are preserved locally; the curated report makes their
aggregate findings available without tracking those large payloads.

The notebook deliberately retains saved outputs, selected source excerpts and
historical local-path metadata. Its early preparation prose describes an earlier
stage; follow the current [README](../README.md) for fresh setup and execution.
The [MIT license](../LICENSE) covers original project code/documentation; it does
not relicense third-party excerpts or dependencies. See the
[third-party notices](../THIRD_PARTY_NOTICES.md).

## Before uploading

Use a Git client that applies this project's `.gitignore`. After creating or
selecting the repository at this directory, inspect its changes before committing:

```bash
git status --short --untracked-files=all
```

Include the hidden `.gitignore` and `.python-version` files. Do not force-add
ignored files or drag the entire local directory into a browser upload: a browser
file upload does not apply `.gitignore` filtering. If using that interface, select
only the included files/directories listed above. Existing tracked files are not
removed by `.gitignore`; review them separately if this folder already belongs to
a repository.

The report's [checksums](../reports/run_001/checksums.json) and
[provenance](../reports/run_001/provenance.json) describe its curated evidence.
Original test and timing receipts describe their recorded execution, not a new
runtime check performed during GitHub preparation.

## Verification scope

Preparation checks file preservation, Python/notebook syntax, JSON parsing,
report checksums and links, plus inclusion/exclusion behavior in an isolated Git
repository. It does not change notebook outputs, run tests, restore the deleted
environment/data, train, evaluate or start a service. Installation/runtime
verification remains a separate explicit action using the README instructions.
