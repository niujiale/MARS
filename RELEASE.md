# Release process

MARS ships formal, citable releases so that a published analysis can name the
exact version of the tool it used. This document is for maintainers.

## Versioning policy

MARS follows [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

- **MAJOR** —incompatible changes to the CLI, to the feature-matrix layout,
  or to the pretrained model interface. A model trained under an older MAJOR
  version is not guaranteed to be applicable.
- **MINOR** —new functionality that does not change existing outputs.
- **PATCH** —bug fixes and documentation only.

The version lives in exactly one place:

```
mars/_version.py  ->  __version__ = "4.0.0"
```

`pyproject.toml` reads it via `[tool.setuptools.dynamic]`, `nfl_py` imports it,
every `--version` flag reports it, and the release workflow refuses to publish
if it disagrees with the git tag. There is no second copy to forget.

## Cutting a release

1. Update `mars/_version.py` to the new version.
2. Move the `## [Unreleased]` items in `CHANGELOG.md` under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading and update the link definitions at the
   bottom of the file.
3. Update `version:` and `date-released:` in `CITATION.cff`.
4. Verify the tree builds and passes its tests in a clean environment:

   ```bash
   bash install.sh                       # or: pip install -e '.[train,dev]'
   mars doctor
   pytest tests/ -v
   bash run_example.sh --clean
   python -m build                       # produces dist/*.tar.gz and dist/*.whl
   twine check dist/*
   ```

5. Commit, tag and push. The tag must be `v` + the version in
   `mars/_version.py`:

   ```bash
   git commit -am "Release v4.0.0"
   git tag -a v4.0.0 -m "MARS 4.0.0"
   git push origin main --follow-tags
   ```

6. Pushing the tag triggers `.github/workflows/release.yml`, which rebuilds
   the distributions, checks the tag against `mars/_version.py`, and creates
   the GitHub Release with the sdist and wheel attached.

7. Confirm the release page lists the artifacts, then verify the published
   archive installs standalone:

   ```bash
   pip install "https://github.com/niujiale/MARS/archive/refs/tags/v4.0.0.tar.gz"
   mars --version
   ```

## Checklist

- [ ] `mars/_version.py` bumped
- [ ] `CHANGELOG.md` section added, links updated
- [ ] `CITATION.cff` version and date updated
- [ ] `pytest tests/ -v` passes
- [ ] `bash run_example.sh --clean` completes end to end
- [ ] `python -m build` and `twine check dist/*` clean
- [ ] the built sdist, unpacked **outside** the source tree, still passes
      `pytest tests/` and runs `bash run_example.sh --clean` end to end. Running
      these from the checkout does not prove it: a file missing from
      `MANIFEST.in` is still on disk there, so the gap only shows up once the
      archive stands alone.
- [ ] tag `vX.Y.Z` pushed, release workflow green
- [ ] release page shows sdist + wheel

## What ships where

| Artifact | Contains | Use case |
|---|---|---|
| GitHub Release source archive (`.tar.gz` / `.zip`) | everything, including `pretrained_models/` and the bundled example | reproducing a published analysis |
| sdist (`mars_nanopore-X.Y.Z.tar.gz`) | source plus `pretrained_models/`, per `MANIFEST.in` | offline installs |
| wheel (`mars_nanopore-X.Y.Z-py3-none-any.whl`) | Python packages only, no models (~60 MB saved) | installing into an environment that already has the models, or that sets `MARS_PRETRAINED_MODELS` |

Because the wheel deliberately omits the ~60 MB of model JSON, a wheel-only
install must point MARS at the models:

```bash
export MARS_PRETRAINED_MODELS=/path/to/MARS/pretrained_models
```

`mars doctor` reports which of the candidate locations was used.
