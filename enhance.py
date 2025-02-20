import torch
import librosa
import numpy as np
import soundfile as sf

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

# Define Generator Model (Same as in train.py)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.conv2(x)

# Load Trained Model
generator = Generator()
generator.load_state_dict(torch.load("models/generator.pth"))
generator.eval()

# Enhance Speech
def enhance_speech(generator, noisy_audio):
    noisy_spec = to_spectrogram(noisy_audio)
    noisy_tensor = torch.tensor(np.abs(noisy_spec)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        enhanced_spec = generator(noisy_tensor).squeeze().numpy()
    return to_waveform(enhanced_spec)

# Load Noisy Speech & Enhance
noisy_speech, _ = load_audio("data/noisy_speech.wav")
enhanced_audio = enhance_speech(generator, noisy_speech)

# Save Enhanced Speech
sf.write("data/enhanced_speech.wav", enhanced_audio, 16000)
print("Enhanced speech saved as 'data/enhanced_speech.wav'")
