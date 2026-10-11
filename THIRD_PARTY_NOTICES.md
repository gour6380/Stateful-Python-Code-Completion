# Third-party data and dependencies

The [MIT license](LICENSE) covers this project's original code and documentation.
It does not relicense dependencies, the Py150 corpus, source repositories in that
corpus, or the research papers cited by the project.

## Dataset acquisition

Data is downloaded explicitly through the notebook from the
[Py150 creator archive](https://www.sri.inf.ethz.ch/py150) and selected using
[Py150 Open manifests at the pinned revision](https://github.com/google-research-datasets/eth_py150_open/tree/864c140cafa7e104948d7c04cba6d1b1b4d483a6).
The preparation code retains only entries whose manifest license is MIT,
Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, CC0-1.0 or Unlicense. A manifest filter
is recorded provenance, not a replacement for upstream notices.

The Git-eligible project contains aggregate evaluation evidence and the original
executed notebook, whose saved outputs include selected source excerpts. The
MIT license does not relicense those third-party excerpts. Downloaded dataset
directories, raw case/prediction files, fitted tokenizers and model weights are
ignored by Git. Consult each upstream repository's license and notice requirements
before redistributing its files or excerpts. Future data acquisition
records URLs, hashes, sizes and eligibility decisions locally.

## Dependencies and research

Dependency versions and distribution hashes are recorded in
[requirements.txt](requirements.txt). Their licenses remain with their respective
authors and distributions. Research foundations and the distinction between this
adaptation and a reproduction are listed in the [README](README.md#research).
