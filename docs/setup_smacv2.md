# SMACv2 Setup (StarCraft II)

The `smacv2` environment requires a StarCraft II installation and the SMAC map packs.
This repo ships a vendored `smacv2` package at `deps/smacv2` and an installer script at
`deps/install_smacv2.sh` (adapted from `deps/smacv2/README.md`).

## Install

From the repository root:

```bash
bash deps/install_smacv2.sh
pip install -e ./deps/smacv2
```

This sets up:

- StarCraft II 4.10 under `deps/StarCraftII`
- SMAC maps under `deps/StarCraftII/Maps/SMAC_Maps`
- An editable install of the vendored `smacv2` package

Then point `SC2PATH` at the installation in your `.env`:

```bash
SC2PATH=/absolute/path/to/repo/deps/StarCraftII
```

## Notes

- The installer downloads large assets with `wget` and unpacks them with `unzip`; both must be on `PATH`.
- The script targets Linux environments.
- For the vendored package's development extras: `pip install -e "./deps/smacv2[dev]"`.

## Smoke test

```bash
python scripts/envs/test_smacv2.py
```
