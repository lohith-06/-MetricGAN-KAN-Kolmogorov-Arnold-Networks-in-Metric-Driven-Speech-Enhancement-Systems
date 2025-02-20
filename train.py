import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
import os

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)

# Load Audio Function
def load_audio(file_path, sr=16000):
    waveform, sample_rate = librosa.load(file_path, sr=sr)
    return waveform, sample_rate

# Convert to Spectrogram
def to_spectrogram(audio, n_fft=512, hop_length=256):
    return librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

# Convert Back to Waveform
def to_waveform(spectrogram, hop_length=256):
    return librosa.istft(spectrogram, hop_length=hop_length)

# Define Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.conv2(x)

# Define Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        return torch.sigmoid(self.conv2(x))

# Knowledge-Aided Constraint (KAN Loss)
def knowledge_constraint(enhanced, clean):
    return torch.mean((enhanced - clean) ** 2)  # Ensures energy consistency

# Initialize Models
generator = Generator()
discriminator = Discriminator()

# Optimizers & Loss
optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)
adversarial_loss = nn.BCELoss()

# Load Data
noisy_speech, sr = load_audio("data/noisy_speech.wav")
clean_speech, sr = load_audio("data/clean_speech.wav")

# Convert to Spectrograms
noisy_spec = to_spectrogram(noisy_speech)
clean_spec = to_spectrogram(clean_speech)

# Convert to Tensors
noisy_tensor = torch.tensor(np.abs(noisy_spec)).unsqueeze(0).unsqueeze(0)
clean_tensor = torch.tensor(np.abs(clean_spec)).unsqueeze(0).unsqueeze(0)

# Training Loop
epochs = 100
for epoch in range(epochs):
    optimizer_G.zero_grad()
    enhanced = generator(noisy_tensor)
    g_loss = knowledge_constraint(enhanced, clean_tensor)
    g_loss.backward()
    optimizer_G.step()

    optimizer_D.zero_grad()
    real_output = discriminator(clean_tensor)
    fake_output = discriminator(enhanced.detach())
    d_loss = adversarial_loss(real_output, torch.ones_like(real_output)) + \
             adversarial_loss(fake_output, torch.zeros_like(fake_output))
    d_loss.backward()
    optimizer_D.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: G Loss={g_loss.item()}, D Loss={d_loss.item()}")

# Save Trained Model
torch.save(generator.state_dict(), "models/generator.pth")
torch.save(discriminator.state_dict(), "models/discriminator.pth")
print("Model saved successfully!")
