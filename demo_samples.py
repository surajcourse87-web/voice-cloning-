import os
import wave
import struct
import numpy as np
from typing import List, Dict, Any

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio_samples")
os.makedirs(AUDIO_DIR, exist_ok=True)

SAMPLE_RATE = 16000

def _write_wav(filename: str, audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Write float32/int16 numpy array to 16-bit PCM WAV."""
    path = os.path.join(AUDIO_DIR, filename)
    # Clip and normalize
    audio_clipped = np.clip(audio_data, -0.99, 0.99)
    int16_data = (audio_clipped * 32767).astype(np.int16)

    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())
    return path

def generate_synthetic_clone_audio(duration: float = 6.0, base_f0: float = 140.0) -> np.ndarray:
    """
    Synthesize speech signal with characteristic Neural Vocoder & AI Clone artifacts:
    - Overly flat F0 pitch contour (no natural vocal jitter)
    - Unnatural phase dispersion in upper frequencies (>4000 Hz)
    - Digital zero-drop silence without soft inhalation/exhalation
    - High-frequency comb filter phase leakage (HiFi-GAN vocoder artifact)
    """
    n_samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    # Syllable cadence envelope (speech-like bursts: ~3.5 words/sec)
    cadence = 0.5 * (1.0 + np.sin(2 * np.pi * 3.2 * t))
    burst_envelope = np.where(cadence > 0.35, 1.0, 0.0)
    # Smooth slightly to avoid loud clicks
    window = np.ones(int(SAMPLE_RATE * 0.02)) / (SAMPLE_RATE * 0.02)
    burst_envelope = np.convolve(burst_envelope, window, mode='same')

    # Flatline F0 (almost 0 jitter, robotic pitch perfection)
    f0 = base_f0
    carrier = np.sin(2 * np.pi * f0 * t)
    # Harmonics (Formant F1=500Hz, F2=1500Hz, F3=2500Hz)
    h1 = 0.7 * np.sin(2 * np.pi * (2 * f0) * t)
    h2 = 0.5 * np.sin(2 * np.pi * (3 * f0) * t)
    h3 = 0.3 * np.sin(2 * np.pi * (4 * f0) * t)
    h4 = 0.2 * np.sin(2 * np.pi * (5 * f0) * t)
    voice_base = carrier + h1 + h2 + h3 + h4

    # Neural Vocoder Artifacts:
    # 1. High-frequency phase dispersion and chirps in 4500-7500 Hz (characteristic of neural vocoder artifacts)
    hf_chirp = 0.35 * np.sin(2 * np.pi * 5400 * t + 12.0 * np.sin(2 * np.pi * 180 * t))
    hf_chirp2 = 0.25 * np.sin(2 * np.pi * 6800 * t + 6.0 * np.cos(2 * np.pi * 220 * t))
    # 2. Add vocoder noise leakage
    vocoder_leakage = 0.12 * np.random.normal(0, 1, n_samples)

    synthetic_signal = (voice_base * burst_envelope) + (hf_chirp * burst_envelope) + (hf_chirp2 * burst_envelope) + (vocoder_leakage * burst_envelope)
    # Normalize
    synthetic_signal = synthetic_signal / (np.max(np.abs(synthetic_signal)) + 1e-6)
    return synthetic_signal

def generate_authentic_human_audio(duration: float = 6.0, base_f0: float = 135.0) -> np.ndarray:
    """
    Synthesize speech signal with natural human biological characteristics:
    - Organic Pitch Jitter (~1.2% cycle-to-cycle micro-tremor)
    - Natural intonation contour & pitch drift
    - Gentle breathing noise during micro-pauses
    - Natural soft glottal onsets
    """
    n_samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Natural conversational cadence with soft inhalations
    cadence = 0.5 * (1.0 + np.sin(2 * np.pi * 2.8 * t + 0.5 * np.sin(2 * np.pi * 0.8 * t)))
    envelope = np.where(cadence > 0.25, cadence, 0.05)  # soft ambient floor
    window = np.ones(int(SAMPLE_RATE * 0.04)) / (SAMPLE_RATE * 0.04)
    envelope = np.convolve(envelope, window, mode='same')

    # Natural pitch micro-tremor (Jitter ~1.5%) + expressive intonation drift
    f0_drift = base_f0 + 12.0 * np.sin(2 * np.pi * 1.5 * t)
    jitter = 2.5 * np.random.normal(0, 1, n_samples)
    f0_inst = f0_drift + jitter
    # Integrate instantaneous frequency to get phase
    phase = 2 * np.pi * np.cumsum(f0_inst) / SAMPLE_RATE

    carrier = np.sin(phase)
    h1 = 0.6 * np.sin(2 * phase)
    h2 = 0.35 * np.sin(3 * phase)
    h3 = 0.15 * np.sin(4 * phase)
    voice_base = carrier + h1 + h2 + h3

    # Biological respiration in micro-pauses (gentle pink-like breath)
    breath_noise = 0.04 * np.random.normal(0, 0.4, n_samples)
    human_signal = (voice_base * envelope) + breath_noise

    # Normalize
    human_signal = human_signal / (np.max(np.abs(human_signal)) + 1e-6)
    return human_signal

DEMO_SCENARIOS = [
    {
        "id": "scam-bank-clone",
        "title": "Bank Security Impersonation (AI Clone Scam)",
        "caller_name": "State Bank Cyber Cell (Spoofed)",
        "caller_number": "+91 98210 44921",
        "category": "SYNTHETIC_CLONE",
        "is_cloned": True,
        "filename": "bank_impersonator_clone.wav",
        "transcript": "Hello, this is officer Vikram from National Bank Security. Your savings account ending in 4109 has been temporarily frozen due to suspicious international attempts. Please verify your OTP to restore access immediately.",
        "description": "ElevenLabs-style clone imitating an authoritative bank officer with flatlined pitch tremor and 5.2kHz vocoder dispersion.",
        "recommended_action": "🚨 KILL CALL & ALERT: Do not disclose OTP. Trigger official bank callback via 1800-helpline."
    },
    {
        "id": "scam-family-emergency",
        "title": "Family Distress / Relative Emergency (AI Clone Scam)",
        "caller_name": "Grandson Rohan (Spoofed ID)",
        "caller_number": "+91 99342 11098",
        "category": "SYNTHETIC_CLONE",
        "is_cloned": True,
        "filename": "relative_emergency_clone.wav",
        "transcript": "Mom, please don't panic! I lost my wallet and phone at the railway station. I'm borrowing a stranger's phone. Please UPI ₹15,000 to this number right now or I won't make it to college!",
        "description": "Voice-conversion clone mimicking emotional distress. High-frequency phase anomalies and lack of physiological respiratory pauses.",
        "recommended_action": "⚠️ SUSPECT CLONE: Call family member back directly on their original known number before sending funds."
    },
    {
        "id": "authentic-manager",
        "title": "Office Colleague / Manager (Authentic Voice)",
        "caller_name": "Vikram Malhotra (Product VP)",
        "caller_number": "+91 98111 87654",
        "category": "AUTHENTIC_HUMAN",
        "is_cloned": False,
        "filename": "authentic_manager.wav",
        "transcript": "Hey Arjun, good morning! Just wanted to check if you had a chance to review the Q3 architecture slide deck before our client demo at 3 PM today.",
        "description": "Genuine human speaker with natural vocal fold jitter (1.4%), realistic breathing micro-pauses, and organic vocal tract formants.",
        "recommended_action": "✅ SAFE: Natural voice markers confirmed. Call verified authentic."
    },
    {
        "id": "authentic-customer-care",
        "title": "Authentic Bank Helpdesk Call",
        "caller_name": "Axis Priority Banking Desk",
        "caller_number": "1800 419 5577",
        "category": "AUTHENTIC_HUMAN",
        "is_cloned": False,
        "filename": "authentic_support.wav",
        "transcript": "Good afternoon, thank you for reaching out to Axis Priority Banking. My name is Priya. I am calling to confirm your appointment scheduled for tomorrow at the Indiranagar branch.",
        "description": "Natural human acoustic signature with organic cadence, natural harmonic ratios, and absence of neural synthesis cues.",
        "recommended_action": "✅ SAFE: Certified genuine human voice."
    }
]

def ensure_demo_audio_files():
    """Generates all demo WAV files on startup if not present."""
    # Bank clone
    bank_path = os.path.join(AUDIO_DIR, "bank_impersonator_clone.wav")
    if not os.path.exists(bank_path):
        sig = generate_synthetic_clone_audio(duration=7.0, base_f0=145.0)
        _write_wav("bank_impersonator_clone.wav", sig)

    # Relative emergency clone
    rel_path = os.path.join(AUDIO_DIR, "relative_emergency_clone.wav")
    if not os.path.exists(rel_path):
        sig = generate_synthetic_clone_audio(duration=6.5, base_f0=175.0)
        _write_wav("relative_emergency_clone.wav", sig)

    # Authentic manager
    mgr_path = os.path.join(AUDIO_DIR, "authentic_manager.wav")
    if not os.path.exists(mgr_path):
        sig = generate_authentic_human_audio(duration=6.0, base_f0=130.0)
        _write_wav("authentic_manager.wav", sig)

    # Authentic support
    sup_path = os.path.join(AUDIO_DIR, "authentic_support.wav")
    if not os.path.exists(sup_path):
        sig = generate_authentic_human_audio(duration=6.5, base_f0=190.0)
        _write_wav("authentic_support.wav", sig)

ensure_demo_audio_files()
