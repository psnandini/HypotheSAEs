# Hippocorpus results

Recalled (label 1) vs. imagined (label 0) stories. 20 SAE neurons selected by correlation; Qwen3-8B as interpreter and annotator.

- `hippocorpus_hypotheses.csv`: training-set neuron correlations, interpretations, and fidelity F1.
- `hippocorpus_holdout_eval.csv`: held-out evaluation (1,101 stories). AUROC 0.594; 0/20 hypotheses significant at p < 0.005.

The Colab runtime reset before download, so these tables were rebuilt from the printed notebook output.

# Drug reviews results

Positive (rating 9-10, label 1) vs. negative (rating 1-4, label 0) Drugs.com medication reviews (UCI Drug Review dataset, CC BY 4.0). Balanced sample of 6,000 de-duplicated reviews. Same SAE, selection, and LLM settings as Hippocorpus.

- `drugs_hypotheses.csv`: training-set neuron correlations, interpretations, and fidelity F1.
- `drugs_holdout_eval.csv`: held-out evaluation (1,200 reviews). AUROC 0.913; 8/20 hypotheses significant at p < 0.005.

# Drug reviews follow-ups

From `notebooks/followups_drug_reviews.ipynb` (run on Modal; SAE retrained with a fixed seed, so neuron numbers differ from the main run).

- `followup_neuron_candidates.csv`: 6 candidate labels for the most predictive neuron (3 with the main run's task instructions, 3 with neutral instructions) and their fidelity scores.
- `followup_neuron_holdout.csv`: held-out test of the main run's "milligrams" label vs. the best new label, each on its own.
- `followup_llm_annotations.csv`: Qwen3-8B yes/no annotations, 1,200 held-out reviews × 20 hypotheses.
- `followup_agreement.csv`: per hypothesis, how well embedding cosine similarity agrees with the LLM annotations.
- `followup_llm_vs_embedding.csv`: regression coefficients and significance under LLM, binarized-embedding, and raw-embedding annotation.
- `followup_summary.csv`: held-out AUROC, number of significant hypotheses, and runtime for each annotation method.
