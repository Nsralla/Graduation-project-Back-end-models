audios folders link:https://drive.google.com/drive/folders/1vY07U0pDacUAGAMRv7tPXqVbiCULFTs_

project idea:
English language proficiency score.
possible scores : A2,B1,B2,C.
process pipeline: Audio -> Wav2vec finetuned feature extractor, Audio-> whisper ->Bert finetuned feature extractor -> Concatenate Bert and Wav2vec features, apply interpolation and send it as input to TCN.

