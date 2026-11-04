# Hippocorpus results

Recalled (label 1) vs. imagined (label 0) stories. 20 SAE neurons selected by correlation; Qwen3-8B as interpreter and annotator.

- `hippocorpus_hypotheses.csv`: training-set neuron correlations, interpretations, and fidelity F1.
- `hippocorpus_holdout_eval.csv`: held-out evaluation (1,101 stories). AUROC 0.594; 0/20 hypotheses significant at p < 0.005.

The Colab runtime reset before download, so these tables were rebuilt from the printed notebook output.

# Drug reviews results

Positive (rating 9-10, label 1) vs. negative (rating 1-4, label 0) Drugs.com medication reviews (UCI Drug Review dataset, CC BY 4.0). Balanced sample of 6,000 de-duplicated reviews. Same SAE, selection, and LLM settings as Hippocorpus.

- `drugs_hypotheses.csv`: training-set neuron correlations, interpretations, and fidelity F1.
- `drugs_holdout_eval.csv`: held-out evaluation (1,200 reviews). AUROC 0.913; 8/20 hypotheses significant at p < 0.005.
