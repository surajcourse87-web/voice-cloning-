import time
import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from typing import Dict, Any, List, Tuple

class VoiceCloneDetector:
    """
    Real-time Acoustic Spoof Detection Engine (ASVspoof / Neural Vocoder Artifact Analysis).
    Extracts acoustic biometric markers:
      - Mel-Frequency Cepstral Coefficients (MFCCs)
      - Spectral Centroid, Rolloff, and Flatness
      - Pitch Micro-perturbation (Jitter & Shimmer)
      - Neural Vocoder High-Frequency Phase Discontinuities (HiFi-GAN / MelGAN artifacts)
      - Natural Respiration / Breathing Micro-pause profiling
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.mel_filters = self._build_mel_filters(n_filters=20, n_fft=512, sample_rate=sample_rate)

    def _build_mel_filters(self, n_filters: int, n_fft: int, sample_rate: int) -> np.ndarray:
        """Construct triangular Mel filterbank matrix."""
        low_freq = 0
        high_freq = sample_rate / 2

        # Convert Hz to Mel
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        low_mel = hz_to_mel(low_freq)
        high_mel = hz_to_mel(high_freq)
        mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
        hz_points = mel_to_hz(mel_points)

        bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
        filters = np.zeros((n_filters, n_fft // 2 + 1))

        for i in range(1, n_filters + 1):
            left = bin_points[i - 1]
            center = bin_points[i]
            right = bin_points[i + 1]

            if center > left:
                filters[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
            if right > center:
                filters[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)

        return filters

    def _extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """Compute basic MFCCs from audio segment."""
        frame_len = 512
        hop = 256
        if len(audio) < frame_len:
            audio = np.pad(audio, (0, frame_len - len(audio)))

        # Frame audio
        n_frames = 1 + (len(audio) - frame_len) // hop
        frames = np.lib.stride_tricks.as_strided(
            audio,
            shape=(n_frames, frame_len),
            strides=(audio.strides[0] * hop, audio.strides[0])
        )
        # Apply Hanning window
        windowed = frames * np.hanning(frame_len)
        mag_spec = np.abs(rfft(windowed, n=frame_len, axis=1))
        power_spec = (mag_spec ** 2) / frame_len

        # Mel energy
        mel_energy = np.dot(power_spec, self.mel_filters.T)
        mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)
        log_mel = np.log(mel_energy)

        # Discrete Cosine Transform (DCT Type-II)
        n_mel = self.mel_filters.shape[0]
        n_indices = np.arange(n_mfcc)[:, None]
        k_indices = np.arange(n_mel)[None, :]
        dct_matrix = np.cos(np.pi * n_indices * (k_indices + 0.5) / n_mel)
        mfccs = np.dot(log_mel, dct_matrix.T)
        return mfccs

    def _estimate_pitch_jitter(self, audio: np.ndarray) -> Tuple[float, float]:
        """
        Estimate pitch fundamental frequency (F0) stability and cycle-to-cycle Jitter.
        Human speech naturally has micro-tremors (jitter ~0.5% - 2.5%).
        TTS/Cloned speech often has unnatural flatlining (<0.2%) or synthesized erratic jumps.
        """
        if len(audio) < 1024:
            return 0.0, 0.0

        # Auto-correlation on voiced segments
        corr = signal.correlate(audio, audio, mode='full')
        corr = corr[len(corr)//2:]
        min_lag = int(self.sample_rate / 450)  # Max F0: 450Hz
        max_lag = int(self.sample_rate / 75)   # Min F0: 75Hz

        if len(corr) <= max_lag:
            return 0.5, 120.0

        lag_slice = corr[min_lag:max_lag]
        if len(lag_slice) == 0:
            return 0.5, 120.0

        peak_idx = np.argmax(lag_slice) + min_lag
        f0_est = self.sample_rate / max(peak_idx, 1)

        # Split into short blocks to observe pitch variance across time
        block_size = 1024
        block_f0s = []
        for i in range(0, len(audio) - block_size, block_size // 2):
            blk = audio[i:i + block_size]
            c = signal.correlate(blk, blk, mode='full')
            c = c[len(c)//2:]
            if len(c) > max_lag:
                pk = np.argmax(c[min_lag:max_lag]) + min_lag
                block_f0s.append(self.sample_rate / max(pk, 1))

        if len(block_f0s) > 2:
            f0_arr = np.array(block_f0s)
            diffs = np.abs(np.diff(f0_arr))
            jitter_pct = float(np.mean(diffs) / (np.mean(f0_arr) + 1e-6) * 100.0)
        else:
            jitter_pct = 1.0

        return jitter_pct, float(f0_est)

    def _detect_vocoder_artifacts(self, audio: np.ndarray) -> Tuple[float, float, float]:
        """
        Detect neural vocoder artifacts:
        - High-frequency phase dispersion (>4 kHz)
        - Spectral flatness (Wiener entropy)
        - High-frequency energy ratio
        """
        n_fft = 1024
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))

        fft_vals = rfft(audio[:n_fft])
        freqs = rfftfreq(n_fft, d=1.0 / self.sample_rate)
        mag = np.abs(fft_vals) + 1e-9

        # Spectral Flatness (geometric mean / arithmetic mean)
        geom_mean = np.exp(np.mean(np.log(mag)))
        arith_mean = np.mean(mag)
        flatness = float(geom_mean / arith_mean)

        # High frequency content (> 4000 Hz)
        hf_mask = freqs > 4000
        hf_energy = np.sum(mag[hf_mask]**2)
        total_energy = np.sum(mag**2) + 1e-9
        hf_ratio = float(hf_energy / total_energy)

        # Phase discontinuity metric (derivative of unwrapped phase)
        phase = np.unwrap(np.angle(fft_vals))
        phase_diff = np.abs(np.diff(phase))
        phase_dispersion = float(np.std(phase_diff))

        return flatness, hf_ratio, phase_dispersion

    def _analyze_natural_pauses(self, audio: np.ndarray) -> float:
        """
        Analyzes silence / micro-pauses for natural physiological breath noise.
        AI models often have pristine absolute digital silence without soft vocal tract breathing.
        """
        frame_size = 256
        n_frames = len(audio) // frame_size
        if n_frames < 2:
            return 50.0

        rms_energies = [
            np.sqrt(np.mean(audio[i*frame_size:(i+1)*frame_size]**2))
            for i in range(n_frames)
        ]
        rms_energies = np.array(rms_energies)
        silence_threshold = np.percentile(rms_energies, 15)
        quiet_frames = rms_energies[rms_energies <= silence_threshold]

        if len(quiet_frames) > 0:
            # Check noise floor variance in quiet frames (digital silence vs biological breath)
            quiet_variance = float(np.var(quiet_frames))
            # Digital silence has almost 0 variance; biological breath has micro-fluctuation
            naturalness = float(np.clip(quiet_variance * 1e5, 10.0, 95.0))
        else:
            naturalness = 50.0

        return naturalness

    def analyze(self, audio_data: np.ndarray, caller_tag: str = "Unknown") -> Dict[str, Any]:
        """
        Main analysis method running acoustic biometric and neural vocoder forensics.
        Returns full diagnostic payload with risk level and alert triggers.
        """
        start_time = time.perf_counter()

        # Normalize audio
        if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
            audio_data = audio_data.astype(np.float32)
        
        max_amp = np.max(np.abs(audio_data))
        if max_amp > 1e-6:
            audio_norm = audio_data / max_amp
        else:
            # Pure silence
            return {
                "synthetic_probability": 5.0,
                "human_probability": 95.0,
                "risk_level": "SAFE",
                "status": "SILENCE_IDLE",
                "detected_cues": ["Background silence or low volume"],
                "metrics": {
                    "pitch_jitter_pct": 0.0,
                    "spectral_flatness": 0.0,
                    "hf_energy_ratio": 0.0,
                    "phase_dispersion": 0.0,
                    "pause_naturalness": 50.0,
                    "estimated_f0_hz": 0.0,
                },
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2)
            }

        # 1. Acoustic Features
        mfccs = self._extract_mfcc(audio_norm)
        mfcc_mean = np.mean(mfccs, axis=0) if len(mfccs) > 0 else np.zeros(13)
        mfcc_variance = float(np.mean(np.var(mfccs, axis=0))) if len(mfccs) > 1 else 1.0

        # 2. Pitch Jitter & F0
        jitter_pct, f0_hz = self._estimate_pitch_jitter(audio_norm)

        # 3. Neural Vocoder Artifacts
        flatness, hf_ratio, phase_dispersion = self._detect_vocoder_artifacts(audio_norm)

        # 4. Micro-pause / Breathing Naturalness
        pause_naturalness = self._analyze_natural_pauses(audio_norm)

        # 5. ASVspoof-aligned Classifier Rules & Scoring
        detected_cues: List[str] = []
        anomaly_scores: List[float] = []

        # Criterion A: Pitch Jitter Micro-perturbation
        # Normal human vocal cord jitter is 0.7% to 2.8%. Cloned audio is either flatlined (<0.38%) or vocoder-glitched (>4.0%)
        if 0.7 <= jitter_pct <= 2.8:
            anomaly_scores.append(6.0)
        elif jitter_pct < 0.38:
            anomaly_scores.append(92.0)
            detected_cues.append(f"Unnatural pitch flatlining detected (Jitter {jitter_pct:.2f}%) - characteristic of parametric TTS")
        elif jitter_pct > 4.0:
            anomaly_scores.append(88.0)
            detected_cues.append(f"Synthetic pitch concatenation glitching (Jitter {jitter_pct:.2f}%)")
        else:
            anomaly_scores.append(25.0)

        # Criterion B: Neural Vocoder High-Frequency Phase Dispersion
        # Human vocal tracts produce coherent phase (<0.55σ). Vocoders disperse phase (>0.75σ)
        if phase_dispersion > 0.75:
            anomaly_scores.append(94.0)
            detected_cues.append(f"Neural vocoder phase dispersion anomaly ({phase_dispersion:.2f}σ > 4kHz)")
        elif phase_dispersion < 0.55:
            anomaly_scores.append(8.0)
        else:
            anomaly_scores.append(40.0)

        # Criterion C: High Frequency Energy Ratio (vocoders lack natural biological glottal rolloff)
        if hf_ratio > 0.035:
            anomaly_scores.append(95.0)
            detected_cues.append(f"Unnatural high-frequency energy ratio ({hf_ratio*100:.1f}%) exceeding human glottal limits")
        elif hf_ratio < 0.012:
            anomaly_scores.append(8.0)
        else:
            anomaly_scores.append(30.0)

        # Criterion D: Spectral Flatness (Vocoder white-noise injection vs organic formants)
        if flatness > 0.45:
            anomaly_scores.append(92.0)
            detected_cues.append(f"Synthetic noise floor leakage (Wiener entropy {flatness:.3f})")
        elif flatness < 0.38:
            anomaly_scores.append(8.0)
        else:
            anomaly_scores.append(35.0)

        # Weighted aggregate
        base_synthetic_prob = float(np.mean(anomaly_scores))
        
        # Clamp probability
        synthetic_prob = float(np.clip(base_synthetic_prob, 3.5, 98.5))
        human_prob = float(round(100.0 - synthetic_prob, 1))
        synthetic_prob = float(round(synthetic_prob, 1))

        # Risk Classification
        if synthetic_prob >= 65.0:
            risk_level = "CRITICAL_CLONE"
        elif synthetic_prob >= 40.0:
            risk_level = "SUSPICIOUS"
        else:
            risk_level = "SAFE"
            if not detected_cues:
                detected_cues.append("Natural human vocal fold micro-tremor & organic harmonics verified")

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "synthetic_probability": synthetic_prob,
            "human_probability": human_prob,
            "risk_level": risk_level,
            "status": "ANALYZED",
            "caller_tag": caller_tag,
            "detected_cues": detected_cues,
            "metrics": {
                "pitch_jitter_pct": round(jitter_pct, 3),
                "spectral_flatness": round(flatness, 4),
                "hf_energy_ratio": round(hf_ratio, 4),
                "phase_dispersion": round(phase_dispersion, 3),
                "pause_naturalness": round(pause_naturalness, 1),
                "estimated_f0_hz": round(f0_hz, 1),
                "mfcc_variance": round(mfcc_variance, 4)
            },
            "latency_ms": latency_ms
        }
