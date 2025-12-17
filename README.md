# Code-and-Data-set-for-COM754

This repository contains the code and processed data used for a quantitative study examining divergence between human and AI-generated empathetic listener responses across emotional contexts.

The original dataset used is the EmpatheticDialogues training set created by Facebook Research. The full dataset could not be uploaded due to size restrictions and is available here:
https://github.com/facebookresearch/EmpatheticDialogues

All analyses were conducted using the training split of this dataset.

The file “THE TEST.csv” is a small sample used only to test and validate the code pipeline and API prompting. It is not the dataset used for the final analysis.

“The Final Dataset.csv” contains the processed dataset used for descriptive statistics and hypothesis testing. This file includes emotional category labels, human listener responses, AI-generated listener responses, and benchmark-derived divergence measures.

The following Python scripts make up the analysis pipeline:

“Filtering.py” assigns emotional valence (positive or negative) and emotional complexity (simple or complex), removes ambiguous emotions, and balances the dataset across the 2×2 emotional framework at the conversation level.

“GPT API Prompter.py” generates AI-based empathetic listener responses using the OpenAI API. Responses are generated using stateless calls, with only prior speaker turns provided as context. Human listener responses and emotion labels are never shown to the model.

“dataset cealing.py” cleans non-semantic artefacts such as placeholder tokens and encoding noise to prevent distortion of benchmark scores.

“Descriptive Stats.py” produces descriptive statistics for the balanced dataset, including conversation counts and divergence summaries.

“Analysis.py” performs hypothesis testing using frequentist ANOVA, testing main effects of emotional complexity and emotional valence, as well as their interaction.

The files “H1_complexity_anova.csv”, “H2_valence_anova.csv”, and “H3_interaction_anova.csv” contain the ANOVA output tables for each hypothesis. The file “hypothesis_results_report.txt” provides a plain-text summary of these results.

This repository is intended for academic assessment and methodological transparency. The study does not evaluate clinical safety, therapeutic effectiveness, or demographic fairness.
