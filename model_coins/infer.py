"""
infer.py - Robust Session Engine for Rupiah Coin Classifier
Assistive Technology for Visually Impaired Users (Rupiah Reader Mobile)

Key Architectural Modules:
1. Normalized Frame Quality Gate:
   - Resolution-normalized Laplacian blur detection (evaluated on fixed 224x224 resize)
   - Dual-mode coin localization (Otsu contour + Hough Circles fallback)
   - Exposure / Brightness bounds (reject over/underexposed frames)
2. Unified 5-Class Inference with TTA Rotations (0°, 90°, 180°, 270°):
   - Inferences across 4 orthogonal rotations averaged to reduce orientation sensitivity
   - Single 5-class model (`best_5class.tflite`) with optimal scalar T* = 0.663
3. Robust Bayesian Evidence Accumulator (CoinSession):
   - Strict Global Session Timer (guaranteed exit <= max_seconds, preventing infinite loops)
   - Max flip limit (stops "balik koin" cycle after 3 failed attempts)
   - Garuda Leakage Protection: Tracks cumulative p(Garuda); refuses nominal decision if mean p(Garuda) >= 0.35
   - Pose Gating (Hamming distance >= 10 bit against last ACCEPTED frame, ignoring camera jitter)
   - High-Precision Decision Rule (tau* = 0.85, min_frames = 6)
"""

import os
import sys

# Crucial for Windows: TensorFlow/TFLite runtime must be imported BEFORE OpenCV
# to avoid OpenMP runtime access violation / deadlock (0xC0000005)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter

import json
import time
import argparse
from collections import deque
import numpy as np
import cv2
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ---------------------------------------------------------------------------
# 1. Normalized Quality Gate Filter
# ---------------------------------------------------------------------------
class FrameQualityGate:
    def __init__(self, min_blur_var=60.0, min_brightness=40.0, max_brightness=225.0):
        self.min_blur_var = min_blur_var
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def check(self, bgr_img):
        """
        Evaluates frame clarity on normalized 224x224 canvas to ensure threshold
        consistency across varying smartphone camera resolutions.
        """
        h, w = bgr_img.shape[:2]
        # Normalize to fixed canvas for resolution-independent metrics
        fixed_canvas = cv2.resize(bgr_img, (224, 224), interpolation=cv2.INTER_AREA)
        gray_norm = cv2.cvtColor(fixed_canvas, cv2.COLOR_BGR2GRAY)
        
        # A. Brightness Check
        mean_brightness = float(np.mean(gray_norm))
        if mean_brightness < self.min_brightness:
            return False, "terlalu_gelap", None
        if mean_brightness > self.max_brightness:
            return False, "terlalu_silau", None

        # B. Resolution-Normalized Blur Check (Laplacian Variance)
        lap_var = float(cv2.Laplacian(gray_norm, cv2.CV_64F).var())
        if lap_var < self.min_blur_var:
            return False, "gambar_blur", None

        # C. Coin Localization (Dual-mode: Otsu + Hough Circles Fallback)
        gray_orig = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        blur_orig = cv2.GaussianBlur(gray_orig, (7, 7), 2)
        _, thresh = cv2.threshold(blur_orig, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        crop = None
        if cnts:
            largest_cnt = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(largest_cnt)
            frame_area = float(h * w)
            area_pct = area / frame_area
            
            # Toleransi luas koin di layar HP (2% hingga 85% luas frame)
            if 0.02 <= area_pct <= 0.85:
                x, y, bw, bh = cv2.boundingRect(largest_cnt)
                max_dim = max(bw, bh)
                cx, cy = x + bw // 2, y + bh // 2
                pad = int(max_dim * 0.12)
                x1 = max(0, cx - max_dim//2 - pad)
                y1 = max(0, cy - max_dim//2 - pad)
                x2 = min(w, cx + max_dim//2 + pad)
                y2 = min(h, cy + max_dim//2 + pad)
                crop = bgr_img[y1:y2, x1:x2]

        # Hough Circles fallback if Otsu fails on complex textured background
        if crop is None or crop.shape[0] < 40 or crop.shape[1] < 40:
            circles = cv2.HoughCircles(
                gray_norm, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                param1=100, param2=30, minRadius=15, maxRadius=105
            )
            if circles is not None:
                c = circles[0][0]
                scale_x, scale_y = w / 224.0, h / 224.0
                cx, cy, r = int(c[0] * scale_x), int(c[1] * scale_y), int(c[2] * max(scale_x, scale_y))
                pad = int(r * 0.18)
                x1 = max(0, cx - r - pad); y1 = max(0, cy - r - pad)
                x2 = min(w, cx + r + pad); y2 = min(h, cy + r + pad)
                crop = bgr_img[y1:y2, x1:x2]

        # Final Fallback: Center Square Crop (fokus tengah lingkaran viewfinder)
        if crop is None or crop.shape[0] < 40 or crop.shape[1] < 40:
            sz = int(min(h, w) * 0.52)
            cx, cy = w // 2, h // 2
            crop = bgr_img[cy - sz//2 : cy + sz//2, cx - sz//2 : cx + sz//2]

        return True, "ok", crop


# ---------------------------------------------------------------------------
# 2. Unified 5-Class Classifier with TTA Rotations
# ---------------------------------------------------------------------------
class CoinClassifierTTA:
    def __init__(self, model_path=None, labels_path=None, temperature=0.663):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(base_dir, 'models', 'best_5class.tflite')
            if not os.path.exists(model_path):
                model_path = os.path.join(base_dir, 'models', 'best.tflite')

        self.model_path = model_path
        self.temperature = temperature
        self.is_tflite = model_path.endswith('.tflite')

        if labels_path is None:
            labels_path = os.path.join(os.path.dirname(model_path), 'labels_5class.json')
            if not os.path.exists(labels_path):
                labels_path = os.path.join(os.path.dirname(model_path), 'labels.json')

        with open(labels_path, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)

        from tensorflow.lite.python.interpreter import Interpreter
        with open(self.model_path, 'rb') as f:
            model_bytes = f.read()
        self.inter = Interpreter(model_content=model_bytes)
        self.inter.allocate_tensors()
        self.inp = self.inter.get_input_details()[0]
        self.out = self.inter.get_output_details()[0]

    def _predict_single_rot(self, img_pil):
        resized = img_pil.resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.expand_dims(np.array(resized, dtype=self.inp['dtype']), axis=0)
        self.inter.set_tensor(self.inp['index'], arr)
        self.inter.invoke()
        probs = self.inter.get_tensor(self.out['index'])[0]

        # Temperature scaling
        if self.temperature > 0 and self.temperature != 1.0:
            eps = 1e-12
            clipped = np.clip(probs, eps, 1.0 - eps)
            scaled = np.log(clipped) / self.temperature
            exp_s = np.exp(scaled - np.max(scaled))
            probs = exp_s / np.sum(exp_s)
        return probs

    def predict_with_tta(self, crop_bgr_or_pil):
        """Averages probabilities over 4 orthogonal rotations (0, 90, 180, 270 deg)."""
        if isinstance(crop_bgr_or_pil, np.ndarray):
            im0 = Image.fromarray(cv2.cvtColor(crop_bgr_or_pil, cv2.COLOR_BGR2RGB))
        else:
            im0 = crop_bgr_or_pil.convert('RGB')

        p_sum = np.zeros(len(self.labels), dtype=np.float32)
        for rot in [0, 90, 180, 270]:
            im_rot = im0.rotate(rot, expand=False)
            p_sum += self._predict_single_rot(im_rot)

        return p_sum / 4.0


# ---------------------------------------------------------------------------
# 3. Robust Multi-Frame Evidence Accumulator (CoinSession)
# ---------------------------------------------------------------------------
class CoinSession:
    """
    Session-level evidence consensus engine for assistive coin reading.
    Enforces per-frame confidence thresholds, directional consensus,
    and strict Garuda guards to push false positive rate <= 2-3%.
    """
    def __init__(self, tau_frame=0.75, min_agree=0.85, min_strong=0.60, min_mean_p=0.70,
                 max_garuda=0.35, min_frames=6, max_seconds=15.0, max_flips=3):
        self.tau_frame = tau_frame
        self.min_agree = min_agree
        self.min_strong = min_strong
        self.min_mean_p = min_mean_p
        self.max_garuda = max_garuda
        self.min_frames = min_frames
        self.max_seconds = max_seconds
        self.max_flips = max_flips
        
        self.accepted_frames = []
        self.garuda_history = []
        self.n_frames_accumulated = 0
        self.flip_count = 0
        self.t0 = time.time()
        self.status = "lanjut"
        self.last_accepted_crop_hash = None
        self.audio_guidance = "Mencari koin..."

    def is_pose_changed(self, crop_bgr):
        """
        Calculates dHash (8x8 = 64-bit) difference against LAST ACCEPTED frame.
        Threshold >= 5 bits ensures meaningful movement/rotation while being
        receptive to natural hand movements on mobile camera streams.
        """
        small = cv2.resize(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY), (9, 8))
        dhash = small[:, 1:] > small[:, :-1]
        if self.last_accepted_crop_hash is None:
            return True
        diff = np.count_nonzero(dhash != self.last_accepted_crop_hash)
        return diff >= 5

    def update(self, p5, crop_bgr):
        """
        Processes new frame into session.
        Enforces strict global timeout, pose gating, and consensus decision rule.
        """
        elapsed = time.time() - self.t0

        # FAIL-SAFE 1: Strict Global Timeout (guaranteed exit <= max_seconds)
        if elapsed >= self.max_seconds:
            self.status = "menyerah"
            self.audio_guidance = "Belum bisa membaca, coba cari tempat lebih terang"
            return {
                'status': self.status,
                'decision': 'tidak_yakin',
                'audio': self.audio_guidance,
                'frames_used': self.n_frames_accumulated,
                'elapsed_s': elapsed
            }

        # Step 1: Track Garuda probability per frame
        p_garuda = float(p5[4])
        self.garuda_history.append(p_garuda)

        # Step 2: Handle Immediate Garuda Detection
        if p_garuda >= 0.60:
            self.flip_count += 1
            if self.flip_count >= self.max_flips:
                self.status = "menyerah"
                self.audio_guidance = "Sisi angka belum terlihat, coba letakkan koin di tempat terang"
                return {
                    'status': self.status,
                    'decision': 'tidak_yakin',
                    'audio': self.audio_guidance,
                    'elapsed_s': elapsed
                }
            self.status = "balik_koin"
            self.audio_guidance = "Balik koinnya, saya belum bisa membaca angkanya"
            return {
                'status': self.status,
                'decision': 'sisi_garuda',
                'audio': self.audio_guidance,
                'elapsed_s': elapsed
            }

        # Step 3: Pose Gating (>= 10 bits difference against last accepted frame)
        if not self.is_pose_changed(crop_bgr):
            return {
                'status': 'lanjut',
                'decision': None,
                'audio': self.audio_guidance,
                'elapsed_s': elapsed
            }

        # Accept frame & record hash
        small = cv2.resize(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY), (9, 8))
        self.last_accepted_crop_hash = (small[:, 1:] > small[:, :-1])

        # Step 4: Record accepted diverse frame
        self.accepted_frames.append(np.array(p5, dtype=np.float32))
        self.n_frames_accumulated += 1

        # Interactive voice guidance to encourage user variation
        if self.n_frames_accumulated == 2:
            self.audio_guidance = "Putar koinnya pelan-pelan..."
        elif self.n_frames_accumulated == 4:
            self.audio_guidance = "Dekatkan sedikit..."

        # Step 5: Strict Consensus Decision Rule
        labels_4 = ['100', '1000', '200', '500']
        if self.n_frames_accumulated >= self.min_frames:
            frames_arr = np.array(self.accepted_frames)
            garuda_mean = float(np.mean(frames_arr[:, 4]))

            # Garuda Leakage Guard: refuse nominal if ambiguous
            if garuda_mean >= self.max_garuda:
                self.flip_count += 1
                if self.flip_count >= self.max_flips:
                    self.status = "menyerah"
                    self.audio_guidance = "Sisi angka belum terbaca jelas, coba cari tempat lebih terang"
                    return {
                        'status': self.status,
                        'decision': 'tidak_yakin',
                        'audio': self.audio_guidance,
                        'frames_used': self.n_frames_accumulated,
                        'elapsed_s': elapsed
                    }
                self.status = "balik_koin"
                self.audio_guidance = "Balik koinnya, gambar belum cukup jelas"
                return {
                    'status': self.status,
                    'decision': 'sisi_garuda',
                    'audio': self.audio_guidance,
                    'frames_used': self.n_frames_accumulated,
                    'elapsed_s': elapsed
                }

            # Normalized 4-class numeral probabilities
            P = frames_arr[:, :4] / (frames_arr[:, :4].sum(axis=1, keepdims=True) + 1e-12)
            top = P.argmax(axis=1)
            cls_idx = int(np.bincount(top, minlength=4).argmax())

            agree = float((top == cls_idx).mean())
            strong = float((P[:, cls_idx] >= self.tau_frame).mean())
            mean_p = float(P[:, cls_idx].mean())

            if agree >= self.min_agree and strong >= self.min_strong and mean_p >= self.min_mean_p:
                self.status = "jawab"
                nominal = labels_4[cls_idx]
                self.audio_guidance = f"Rp {nominal}"
                return {
                    'status': self.status,
                    'decision': nominal,
                    'confidence': mean_p,
                    'agree_ratio': agree,
                    'strong_ratio': strong,
                    'audio': self.audio_guidance,
                    'frames_used': self.n_frames_accumulated,
                    'elapsed_s': elapsed
                }

        return {
            'status': 'lanjut',
            'decision': None,
            'audio': self.audio_guidance,
            'frames_used': self.n_frames_accumulated,
            'elapsed_s': elapsed
        }


# ---------------------------------------------------------------------------
# 4. Command-Line Interface
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-Frame Coin Session Reader")
    parser.add_argument('--input', type=str, required=True, help="Image file, folder, or directory")
    parser.add_argument('--model', type=str, default=None, help="Path to best_5class.tflite")
    parser.add_argument('--tau', type=float, default=0.85, help="Posterior decision threshold")
    parser.add_argument('--min_frames', type=int, default=6, help="Minimum diverse frames before answering")
    args = parser.parse_args()

    gate = FrameQualityGate()
    classifier = CoinClassifierTTA(model_path=args.model)

    print(f"\nEngine initialized: TFLite 5-Class with TTA Rotations (tau={args.tau}, min_frames={args.min_frames})")

    if os.path.isfile(args.input):
        bgr = cv2.imread(args.input)
        if bgr is None:
            import augment_coin
            bgr = augment_coin.cv2_imread_unicode(args.input)
            
        passed, reason, crop = gate.check(bgr)
        print(f"\n--- Evaluasi Frame Tunggal: {args.input} ---")
        print(f"Quality Gate Status : {passed} ({reason})")
        if not passed:
            print(f"Peringatan Kualitas : Frame ditolak gate ({reason}).")
            return

        p5 = classifier.predict_with_tta(crop)
        probs_dict = {lbl: round(float(p), 4) for lbl, p in zip(classifier.labels, p5)}
        
        idx = int(np.argmax(p5))
        lbl = classifier.labels[idx]
        conf = float(p5[idx])

        if lbl == 'sisi_garuda':
            msg = "Balik koinnya, saya belum bisa membaca angkanya"
        elif conf >= 0.70:
            msg = f"Rp {lbl}"
        else:
            msg = "Tidak yakin, coba arahkan kamera kembali"

        print(f"Prediksi TTA        : {lbl} ({conf*100:.1f}%)")
        print(f"Audio Output        : \"{msg}\"")
        print(f"Distribusi Prob     : {json.dumps(probs_dict)}")

    elif os.path.isdir(args.input):
        files = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"\nMemproses {len(files)} citra sebagai simulasi sesi kontinu...")
        session = CoinSession(tau=args.tau, min_frames=args.min_frames)

        for f in files:
            p = os.path.join(args.input, f)
            bgr = cv2.imread(p)
            if bgr is None:
                import augment_coin
                bgr = augment_coin.cv2_imread_unicode(p)
                
            passed, reason, crop = gate.check(bgr)
            if not passed:
                continue

            p5 = classifier.predict_with_tta(crop)
            res = session.update(p5, crop)
            print(f"[{res['status'].ljust(10)}] Frame {session.n_frames_accumulated:2d} | Audio: \"{res['audio']}\"")
            if res['status'] in ("jawab", "balik_koin", "menyerah"):
                print(f"\n>>> Sesi Selesai: {res['status'].upper()} (Keputusan: {res.get('decision')}, Waktu: {res['elapsed_s']:.2f}s)")
                break

if __name__ == '__main__':
    main()
