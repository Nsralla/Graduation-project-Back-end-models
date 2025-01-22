from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import os
from fastapi.middleware.cors import CORSMiddleware
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from transformers import BertTokenizer, BertModel
import whisper
from torch.nn import AvgPool1d
import torch.nn.functional as F
from tcn import TCN

torchaudio.set_audio_backend("soundfile")

app = FastAPI()
# Allow requests from React dev server at localhost:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # <--- This is key
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global or at startup: load all large models only once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"used device: {device}")
print("[DEBUG] Loading Wav2Vec2 model/processor...")
wav2vec_checkpoint_path = r'wav2vec\epoch_9'
wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_checkpoint_path)
wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained(wav2vec_checkpoint_path).to(device)

print("[DEBUG] Loading Whisper model...")
model_size = "medium"
whisper_model = whisper.load_model(model_size, device=device)

print("[DEBUG] Loading BERT model/tokenizer...")
bert_checkpoint_path = r'bert\bert_model\bert_finetuned_epoch_{epoch + 1}'
tokenizer = BertTokenizer.from_pretrained(bert_checkpoint_path)
bert_model = BertModel.from_pretrained(bert_checkpoint_path).to(device)

print("[DEBUG] Loading TCN model...")
saved_model_path = r"tcn\trained_tcn_model.pth"
input_size = 1536  # 768 + 768 from your pipeline
num_channels = [64, 128, 256]
num_classes = 4
loaded_tcn_model = TCN(
    num_inputs=input_size,
    num_channels=num_channels,
    num_classes=num_classes,
    kernel_size=3,
    dropout=0.2
)
loaded_tcn_model.load_state_dict(torch.load(saved_model_path, map_location=device))
loaded_tcn_model.to(device)
loaded_tcn_model.eval()

# Your label list
sorted_labels = ["A2", "B1", "B2", "C"]

def extract_audio_features(audio_path):
    # (Same as your code)
    audio_input, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_input = resampler(audio_input)
    if audio_input.shape[0] > 1:  # stereo to mono
        audio_input = torch.mean(audio_input, dim=0, keepdim=True)

    input_values = wav2vec_processor(audio_input.squeeze().numpy(), sampling_rate=16000).input_values[0]
    input_tensor = torch.tensor([input_values], dtype=torch.float).to(device)
    with torch.no_grad():
        outputs = wav2vec_model(input_values=input_tensor, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
    audio_features = hidden_states[0].cpu().numpy()
    return audio_features

def transcribe_audio(audio_path):
    # (Same as your code)
    with torch.no_grad():
        result = whisper_model.transcribe(audio_path, language="en")
        transcription_text = result["text"]
    return transcription_text

def extract_text_features(text):
    # (Same as your code)
    encoded = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
    return token_embeddings.squeeze(0)

def average_pool_features(feature, window_size_ms=200, stride_size_ms=100, feature_stride_ms=20):
    # (Same as your code)
    window_size = window_size_ms // feature_stride_ms
    stride_size = stride_size_ms // feature_stride_ms
    avg_pool = AvgPool1d(kernel_size=window_size, stride=stride_size)
    feature_tensor = torch.tensor(feature, dtype=torch.float)
    feature_tensor = feature_tensor.permute(1, 0).unsqueeze(0)
    pooled_tensor = avg_pool(feature_tensor)
    pooled_feature = pooled_tensor.squeeze(0).permute(1, 0)
    return pooled_feature

def normalize_features(features: torch.Tensor) -> torch.Tensor:
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    std[std == 0] = 1.0
    normalized = (features - mean) / std
    return normalized, mean, std

def denormalize_features(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return features * std + mean

def smooth_features(features: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size <= 1:
        return features
    features_3d = features.transpose(0, 1).unsqueeze(0)
    kernel = torch.ones((features_3d.shape[1], 1, kernel_size), dtype=features.dtype, device=features.device) / kernel_size
    padding = (kernel_size // 2, kernel_size // 2)
    smoothed_3d = F.conv1d(F.pad(features_3d, (padding[0], padding[1]), mode='replicate'), kernel, groups=features_3d.shape[1])
    smoothed = smoothed_3d.squeeze(0).transpose(0, 1)
    return smoothed

def interpolate_features(features: torch.Tensor, target_length: int) -> torch.Tensor:
    T, D = features.shape
    if T == target_length:
        return features
    normalized, mean, std = normalize_features(features)
    features_3d = normalized.unsqueeze(0).transpose(1, 2)
    interpolated_3d = F.interpolate(features_3d, size=target_length, mode='linear', align_corners=True)
    interpolated_normalized = interpolated_3d.squeeze(0).transpose(0, 1)
    interpolated = denormalize_features(interpolated_normalized, mean, std)
    interpolated = smooth_features(interpolated, kernel_size=5)
    return interpolated

@app.post("/predict_audio")
async def predict_audio(file: UploadFile = File(...)):
    """
    This endpoint receives an audio file, saves it to a temporary path, 
    runs the pipeline to produce a TCN-based label, then returns the label.
    """
    try:
        # 1) Save the uploaded file to a temp location
        temp_filename = "temp_upload_audio.wav"
        with open(temp_filename, "wb") as f:
            f.write(await file.read())
            
        if not os.path.exists(temp_filename):
            raise ValueError("Temporary file does not exist.")
        if os.path.getsize(temp_filename) == 0:
            raise ValueError("Temporary file is empty.")
        
        # 2) Extract features (Wav2Vec + average pooling)
        audio_features = extract_audio_features(temp_filename)
        audio_features_averaged = average_pool_features(audio_features)

        # 3) Transcribe audio using Whisper
        transcription = transcribe_audio(temp_filename)

        # 4) Extract text features using BERT
        text_features = extract_text_features(transcription)

        # 5) Interpolate and combine
        T_bert, D_bert = text_features.shape
        T_wav, D_wav = audio_features_averaged.shape
        T_final = (T_bert + T_wav) // 2

        bert_features_interpolated = interpolate_features(text_features, T_final).to(device)
        wav_features_interpolated  = interpolate_features(audio_features_averaged, T_final).to(device)

        combined_features = torch.cat((bert_features_interpolated, wav_features_interpolated), dim=1)  # [T, 1536]
        combined_features = combined_features.unsqueeze(0).permute(0, 2, 1).to(device)  # [1, 1536, T_final]

        # 6) TCN inference
        with torch.no_grad():
            output = loaded_tcn_model(combined_features)  # [1, num_classes]
            predicted_class_id = torch.argmax(output, dim=1).item()

        predicted_label = sorted_labels[predicted_class_id]
        
        # 7) Return the predicted label
        return JSONResponse(content={
            "predicted_label": predicted_label,
            "transcription": transcription
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        # Clean up (remove the temp file)
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

