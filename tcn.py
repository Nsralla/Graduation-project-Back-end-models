import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

##################################
# Utility Functions
##################################

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

##################################
# Temporal Convolutional Network (TCN) Implementation
##################################

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size=5, dilation=1, dropout=0.2):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2  # Maintain sequence length

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, dilation=dilation, padding=padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, dilation=dilation, padding=padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.relu2,
            self.dropout2
        )

        # For residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes=4, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size, dropout=dropout),
                nn.AvgPool1d(kernel_size=2, stride=2)
            ]

        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x, return_features=False):
        # x: [batch_size, num_inputs, seq_len]
        x = self.network(x)
        # Global average pooling to produce [batch_size, num_channels[-1]]
        x = self.global_pool(x).squeeze(-1)
        logits = self.fc(x)
        if return_features:
            return x, logits
        else:
            return logits

##################################
# Dataset and DataLoader
##################################

class CombinedFeaturesDataset(Dataset):
    def __init__(self, feature_dir, valid_labels):
        """
        Args:
            feature_dir: Directory containing .npz files with combined_features.
            valid_labels: Set of valid labels, e.g. {}.
        """
        self.feature_dir = feature_dir
        self.valid_labels = {lbl.lower(): lbl for lbl in valid_labels}
        self.features = []
        self.labels = []
        self.filenames = []
        self._load_features()

    def _load_features(self):
        print(f"Loading features from {self.feature_dir}")
        all_files = [f for f in os.listdir(self.feature_dir) if f.endswith('.npz')]
        for file in tqdm(all_files, desc=f'Loading {os.path.basename(self.feature_dir)}'):
            filepath = os.path.join(self.feature_dir, file)
            try:
                data = np.load(filepath)
                combined_features = data['combined_features']  # shape: [T_final, D_total]
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue

            # Extract label from filename by direct substring match
            label = None
            filename_lower = file.lower()
            for possible_label in valid_labels:
                if possible_label.lower() in filename_lower:
                    label = possible_label
                    break


            if label is not None:
                self.features.append(combined_features)
                self.labels.append(label)
                self.filenames.append(file)
            else:
                print(f"Warning: No valid label found in filename '{file}'")

        print(f"Total valid samples loaded: {len(self.features)}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]  # [T_final, D_total]
        label = self.labels[idx]
        filename = self.filenames[idx]
        # Transpose to [D_total, T_final] for TCN input (channels-first)
        feature_tensor = torch.tensor(feature, dtype=torch.float).transpose(0, 1)
        return feature_tensor, label, filename

def collate_fn(batch):
    # batch: list of (feature_tensor, label_str, filename)
    features, labels, filenames = zip(*batch)
    seq_lengths = [f.shape[1] for f in features]
    max_seq_len = max(seq_lengths)
    hidden_size = features[0].shape[0]

    padded_features = torch.zeros(len(features), hidden_size, max_seq_len)
    for i, f in enumerate(features):
        seq_len = f.shape[1]
        padded_features[i, :, :seq_len] = f

    return padded_features, list(labels), filenames

def encode_labels(labels, valid_labels):
    sorted_labels = sorted(list(valid_labels))
    label_to_id = {lbl: i for i, lbl in enumerate(sorted_labels)}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}
    encoded = [label_to_id[lbl] for lbl in labels]
    return label_to_id, id_to_label, encoded

def prepare_dataloader(feature_dir, valid_labels, batch_size=16, shuffle=False):
    dataset = CombinedFeaturesDataset(feature_dir, valid_labels)
    label_to_id, id_to_label, encoded_labels = encode_labels(dataset.labels, valid_labels)
    dataset.labels = encoded_labels
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return loader, label_to_id, id_to_label

##################################
# Training and Evaluation
##################################

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for features, labels, filenames in train_loader:
        features = features.to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

def evaluate_model(model, loader, criterion, device, id_to_label):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels, filenames in loader:
            features = features.to(device)
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

            logits = model(features)
            loss = criterion(logits, labels_tensor)
            total_loss += loss.item() * features.size(0)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    cls_report = classification_report(all_labels, all_preds, target_names=[id_to_label[i] for i in sorted(id_to_label.keys())])
    return avg_loss, acc, cls_report, all_labels, all_preds

##################################
# Main
##################################

if __name__ == "__main__":
    # Directories with combined interpolated features
    train_combined_dir = r"C:\Users\nsrha\OneDrive\Desktop\Graduation project\interpolation\training"  # Change this to your training directory path
    test_combined_dir = r"C:\Users\nsrha\OneDrive\Desktop\Graduation project\interpolation\testing"    # Change this to your testing directory path

    valid_labels = {"A2", "B1","B2", "C"}

    train_loader, label_to_id, id_to_label = prepare_dataloader(train_combined_dir, valid_labels, batch_size=8, shuffle=True)
    test_loader, _, _ = prepare_dataloader(test_combined_dir, valid_labels, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine input size from a batch
    example_features, example_labels, example_filenames = next(iter(train_loader))
    input_size = example_features.shape[1]

    # TCN parameters
    num_channels = [64, 128,256]  # Example configuration
    num_classes = len(valid_labels)  # should be 3
    model = TCN(num_inputs=input_size, num_channels=num_channels, num_classes=num_classes, kernel_size=7, dropout=0.25)
    model.to(device)

    # Training parameters
    lr = 1e-3
    epochs = 7
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, cls_report, all_labels, all_preds = evaluate_model(model, test_loader, criterion, device, id_to_label)

        print(f"Epoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print("Classification Report:\n", cls_report)
        print("-" * 50)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=[id_to_label[i] for i in sorted(id_to_label.keys())],
                yticklabels=[id_to_label[i] for i in sorted(id_to_label.keys())], cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    ##################################
    # Extract Test Features and Plot with T-SNE
    ##################################
    model.eval()
    test_features_list = []
    test_labels_list = []

    with torch.no_grad():
        for features, labels, filenames in test_loader:
            features = features.to(device)
            feats, _ = model(features, return_features=True)
            test_features_list.append(feats.cpu().numpy())
            test_labels_list.extend(labels)

    test_features_all = np.concatenate(test_features_list, axis=0)
    tsne = TSNE(n_components=2, random_state=42)
    test_features_2d = tsne.fit_transform(test_features_all)

    # Map integer labels back to their string names for plotting
    label_names = [id_to_label[l] for l in test_labels_list]

    # Assign a color to each label
    unique_labels = sorted(list(valid_labels))
    palette = sns.color_palette("hls", len(unique_labels))
    label_to_color = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}
    colors = [label_to_color[lbl] for lbl in label_names]

    plt.figure(figsize=(8,8))
    for lbl in unique_labels:
        idxs = [i for i, x in enumerate(label_names) if x == lbl]
        plt.scatter(test_features_2d[idxs,0], test_features_2d[idxs,1], c=[label_to_color[lbl]], label=lbl, alpha=0.6)

    plt.title("T-SNE visualization of test feature embeddings")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.legend()
    plt.show()

    ##################################
    # Save the Trained Model
    ##################################
    save_path = "trained_tcn_model_icnale_max_corners_corners.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Trained TCN model saved to {save_path}")