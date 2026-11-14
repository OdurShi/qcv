# CLEVR attribution

The files `clevr_val_calibration.jsonl.gz` and `clevr_val_test.jsonl.gz` in this
directory are derived from CLEVR v1.0.

* Dataset: CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual
  Reasoning, v1.0 (J. Johnson, B. Hariharan, L. van der Maaten, L. Fei-Fei, C. L. Zitnick,
  R. Girshick, 2017)
* Source: https://cs.stanford.edu/people/jcjohns/clevr/
* Copyright notice of the dataset: "CLEVR (c) 2017, Facebook, Inc."
* Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/;
  the licence text is in `LICENSE.txt`.

Changes made: the questions of the val split that the grammar `qcv/grammars/clevr.yaml`
compiles were selected (15723 of 149991); each was converted to one JSON line with the
question, its answer, the image file name, and answers of related questions (object counts and
existence) computed from the CLEVR scene graphs and functional programs; the records were
divided into two files. Images are not included.
