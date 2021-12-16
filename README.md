# cnn4epi
CNN model for Enhancer-Promoter Classification

```bash
python3 prepare_data.py --balanced
python3 main.py --cell_line='GM12878' --cross_cell_line='K562'
```

:warning: By default `--seed=42`.

:warning: Unset `--cross_cell_line` for testing on the same cell-line.

## References

**Zeng et al. (2021)**: https://www.frontiersin.org/articles/10.3389/fgene.2021.681259/full

**TargetFinder:** https://github.com/shwhalen/targetfinder
