# AI-Based English Proficiency Scoring

This project evaluates English language proficiency using AI models. It processes spoken English audio and assigns a proficiency score based on extracted features and text analysis. 



This project is a FastAPI-based system that evaluates English language proficiency from speech recordings. It processes audio, extracts linguistic and acoustic features, and assigns a proficiency score using deep learning models.

The system uses:
- **Whisper** to convert speech to text.
- **Wav2Vec** to extract deep audio features.
- **BERT** for linguistic analysis.
- **Temporal Convolutional Network (TCN)** for scoring proficiency.

Features

Speech-to-text conversion with OpenAI's Whisper.

Deep audio feature extraction using Wav2Vec.

Text analysis for fluency and coherence using BERT.

Proficiency classification into levels (A2, B1, B2, C) using TCN.

REST API powered by FastAPI for easy integration.


audios folders used to train the model link:https://drive.google.com/drive/folders/1vY07U0pDacUAGAMRv7tPXqVbiCULFTs_

