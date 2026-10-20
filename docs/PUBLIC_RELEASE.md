# Public release

This repository starts with a clean source snapshot. It contains the frozen protocols, original numerical summaries, orchestration, and the complete tracked evaluation implementation. No private repository history is included.

- Article: https://arxiv.org/abs/2610.24029
- Artifacts and SHA-256 manifests: https://drive.google.com/drive/folders/1eS3xhP3qB_QBep2mUcOfbvddVSj4qHy2?usp=drive_link
- Project page: https://parsa-ra.github.io/lmbehave/
- Website source: https://github.com/parsa-ra/lmbehave-page
- Bundled evaluation revision: `2f121d64e804b83fe8724981fd180d1d5a4c3047`; the original MIT license is retained in `lmms-eval/LICENSE`.

`repos.lock.json` separates bundled source provenance from external checkouts. Existing artifact reports retain their original source hashes and recorded compute paths; these are provenance, not dependencies on private Git history. Use `scripts/artifact_path_remap/` to restore the artifacts elsewhere.

The Pages workflow in this repository checks out the public website source and publishes it at `/lmbehave/`. Website edits can be published immediately with `gh workflow run pages.yml --repo parsa-ra/lmbehave`; a daily scheduled run also refreshes it. The page itself remains buildless HTML, CSS, and JavaScript.

Historical specialized launchers retain their frozen source-revision gates. The current public pipeline is the manifest CLI documented in README.md; historical table reproduction uses the released artifacts and summarizers.
