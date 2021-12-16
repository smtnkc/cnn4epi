#!/bin/bash
for CELL_LINE in 'GM12878' 'HUVEC' 'HeLa-S3' 'K562'; do
    for CROSS_CELL_LINE in 'GM12878' 'HUVEC' 'HeLa-S3' 'K562'; do
        echo python3 main.py --cell_line="$CELL_LINE" --cross_cell_line="$CROSS_CELL_LINE"
        python3 main.py --cell_line="$CELL_LINE" --cross_cell_line="$CROSS_CELL_LINE"
    done
done

echo python3 main.py --cell_line="combined"
python3 main.py --cell_line="combined"
