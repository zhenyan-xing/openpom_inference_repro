# Checkpoints

Place the real OpenPOM single-checkpoint weights here when available:

```text
checkpoints/openpom_experiments_1_checkpoint2.pt
```

If the checkpoint is stored elsewhere, pass it explicitly:

```bash
python scripts/export_openpom_reference.py --checkpoint /path/to/checkpoint2.pt
```

The 133-byte files currently present under `/home/xing/openpom/models/ensemble_models`
are Git LFS pointer files, not usable model weights.
