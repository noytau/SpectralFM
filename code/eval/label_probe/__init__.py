"""
label_probe: few-shot / label-efficiency probing of the SSL backbone's
embedding versus raw input, for a scalar regression label.

Entry point: `python -m eval.label_probe --checkpoint <path> --data <dir>`
(see __main__.py). Fairseq-free — only torch/transformers/sklearn/numpy.
"""
