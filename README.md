# AI-Based English Proficiency Scoring

This project evaluates English language proficiency using AI models. It processes spoken English audio and assigns a proficiency score based on extracted features and text analysis.

## Overview

This FastAPI-based system analyzes speech recordings to evaluate English language proficiency. The system processes audio, extracts both linguistic and acoustic features, and assigns a proficiency score using deep learning models. It leverages the following technologies:

- **Whisper** to convert speech to text.
- **Wav2Vec** to extract deep audio features.
- **BERT** for linguistic analysis.
- **Temporal Convolutional Network (TCN)** for scoring proficiency.

## Features

- **Speech-to-Text Conversion:** Utilizes OpenAI's Whisper for converting audio to text.
- **Deep Audio Feature Extraction:** Uses Wav2Vec to derive robust audio features.
- **Linguistic Analysis:** Employs BERT to evaluate fluency and coherence.
- **Proficiency Classification:** Classifies proficiency into levels (A2, B1, B2, C) with TCN.
- **REST API Integration:** Built with FastAPI for seamless integration with other applications.

> **Note:** The audio folders used to train the model can be accessed [here](https://drive.google.com/drive/folders/1vY07U0pDacUAGAMRv7tPXqVbiCULFTs_).

## Installation

### Prerequisites

- **Python 3.8+**
- **PyTorch**
- **FastAPI**
- **Hugging Face Transformers**
- **torchaudio**
- **OpenAI Whisper**
- **Uvicorn** (for running the FastAPI server)

### Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Nsralla/Graduation-project-Back-end-models.git
   cd Graduation-project-Back-end-models
