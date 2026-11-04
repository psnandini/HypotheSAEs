# Hippocorpus results

Recalled (label 1) vs. imagined (label 0) stories. 20 SAE neurons selected by correlation; Qwen3-8B as interpreter and annotator.

- `hippocorpus_hypotheses.csv`: training-set neuron correlations, interpretations, and fidelity F1.
- `hippocorpus_holdout_eval.csv`: held-out evaluation (1,101 stories). AUROC 0.594; 0/20 hypotheses significant at p < 0.005.

The Colab runtime reset before download, so these tables were rebuilt from the printed notebook output.
