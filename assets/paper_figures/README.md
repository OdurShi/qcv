# Paper figures

Figures of "Questions as Codes: Correcting Multimodal Reasoning at Test Time", cropped from the
manuscript PDF without their captions. The captions below are those printed in the paper.
Regenerate the images with `python scripts/extract_paper_figures.py /path/to/paper.pdf`
(requires PyMuPDF).

| File | Figure | Caption |
|---|---|---|
| `fig01.png` | Figure 1 | From Repeated Computation to Complementary Evidence. Under the same test-time budget, QCV converts repeated model calls into more independent and failure-resilient evidence (top), yielding consistent accuracy gains across all 16 backbone–dataset combinations (bottom). |
| `fig02.png` | Figure 2 | Failure geometry by dataset across test-time budgets. Each panel contains six model–relation settings. Thin paths show individual settings and haloed paths show their medians; marker shape identifies the VLM and area encodes the budget. Repeats are gray and hollow, whereas probes are colored and filled. Arrows connect the B = 9 medians, whose (ρ̄_B, S_B) values appear in the boxes. |
| `fig03.png` | Figure 3 | QCV pipeline on a counting-difference example. The compiler derives constraint-linked auxiliary questions from the target. QCV selects a fixed-budget subset in a calibrated residual geometry, scores complete answer distributions with the frozen VLM, and projects them onto the feasible code to return ŷ_0 = 3. |
| `fig04.png` | Figure 4 | Target separation and empirical correction geometry. Hexagons bin examples by target separation and whitened residual radius; the dashed line marks κ = 1. Heatmaps report final correctness (top) and correction of initial errors (bottom), with outcome contours overlaid. |
| `fig05.png` | Figure 5 | Failure-geometry phase portraits across test-time budgets. Thin paths are backbone–relation-family configurations and haloed paths are dataset medians; arrows indicate B = 3 → 5 → 7 → 9, with QCV endpoints labeled. Bubble area encodes correction rate. Gray guides show B_eff at B = 5 as a common reference; each observation uses its own budget when computing B_eff. The preferred region is upper-left. |
