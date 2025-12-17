# Code-and-Data-set-for-COM754

Overview

This repository contains the full data and code pipeline for a quantitative study examining whether AI-generated empathetic listener responses align with human listener responses across different emotional contexts.

The study uses the EmpatheticDialogues dataset, where emotional context is explicitly elicited from speakers, and introduces AI-generated listener responses under matched conversational conditions. Divergence is evaluated across sentiment, empathy, and emotion detection benchmarks, aggregated into a standardised composite measure.

The primary focus is contextual bias: whether AI responses systematically diverge from human empathy in positive vs negative and simple vs complex emotional settings.

Repository Structure
├── data/
│   ├── ED dataset.csv
│   ├── PreFilteredDataset.csv
│   ├── The Final Dataset.csv
│   └── THE TEST.csv
│
├── code/
│   ├── Filtering.py
│   ├── GPT API Prompter.py
│   ├── dataset cealing.py
│   ├── Descriptive Stats.py
│   └── Analysis.py
│
└── README.md

Datasets
ED dataset.csv

Raw EmpatheticDialogues training data (~25,000 conversations).
Each conversation includes:

A speaker-elicited emotion

A situation description

Human listener responses

Purpose:
Provides human ground truth where emotional context is consciously selected by speakers rather than inferred post hoc.

PreFilteredDataset.csv

Filtered subset of the ED dataset with:

Ambiguous emotions removed

Only emotions clearly classifiable by valence and complexity retained

Purpose:
Improves construct validity prior to AI generation.

The Final Dataset.csv

Main analysis dataset containing:

Human listener responses

AI-generated listener responses

Emotional category labels

Benchmark divergence scores

Standardised composite outcome

Purpose:
Used for all descriptive and inferential analyses.

THE TEST.csv

Small validation subset.

Purpose:
Used to test API calls, prompt logic, and data integrity before full-scale generation.

Code Description
Filtering.py

Assigns each conversation to a 2 × 2 emotional framework:

Valence: Positive / Negative

Complexity: Simple (ID) / Complex (OOD)

Balances the dataset at the conversation level using stratified sampling.

Purpose:
Prevents confounding from unequal category representation.

GPT API Prompter.py

Generates AI empathetic listener responses using GPT-4.1 Mini.

Key constraints:

Stateless API calls (no cross-conversation memory)

Only prior speaker turns provided as context

Human listener responses never shown to the model

Responses constrained to brief, empathetic replies

Purpose:
Ensures fair, matched comparison between human and AI responses.

dataset cealing.py

Cleans non-semantic artefacts including:

EmpatheticDialogues placeholder tokens (e.g. _comma_)

Encoding noise

Excess whitespace

Purpose:
Prevents benchmark distortion due to NLP artefacts.

Descriptive Stats.py

Produces:

Conversation and utterance counts per emotional category

Distribution summaries

Preliminary checks of divergence scores

Purpose:
Transparency and sanity-checking before hypothesis testing.

Analysis.py

Performs hypothesis testing using frequentist ANOVA:

Main effect of emotional complexity

Main effect of emotional valence

Complexity × valence interaction

Effect sizes are reported using η².

Purpose:
Formal statistical evaluation of AI–human divergence.

Methodological Metadata

Design: Quantitative, secondary-data, 2 × 2 factorial

Unit of analysis: Listener responses

Emotional context: Speaker-elicited (not post hoc inferred)

Benchmarks: Sentiment, empathy, emotion detection

Outcome: Standardised composite divergence score

AI model: GPT-4.1 Mini

Balancing: Conversation-level stratified sampling

Notes for Reproducibility

API calls require an OpenAI API key

Test runs can be executed using THE TEST.csv

All processing steps are modular and sequential

No demographic or clinical claims are made
