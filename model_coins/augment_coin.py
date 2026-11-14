"""
augment_coin.py - Synthetic Coin Augmentation & Extraction Module
Designed for Rupiah Coin Classification on Mobile Devices (Assistive Tech).

Strict Rules & Best Practices:
- NO horizontal or vertical flipping (preserves coin numerals).
- Preserves characteristic metal hues (aluminum vs brass/nickel).
- Full 0-360 degree rotation, realistic perspective tilts, glare, noise, blur.
- Finger occlusion restricted strictly to OUTER EDGES (<=15% area), NEVER center.
- Invariant scale computed from coin diameter BEFORE rotation.
- Real crop mixing (25%) to eliminate synthetic-to-real domain gap.
- Live sequence generator (`LiveCoinDataset` via `tf.keras.utils.PyDataset`).
- Robust Unicode path handling for Windows file systems.
"""

import os
import math
import random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

def cv2_imread_unicode(path):
    """Safely loads an image as BGR from a path that may contain non-ASCII unicode characters."""
    if isinstance(path, Path):
        path = str(path)
    with open(path, 'rb') as f:
        bytes_data = bytearray(f.read())
    nparr = np.asarray(bytes_data, dtype=np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def cv2_imwrite_unicode(path, img, params=None):
    """Safely writes an image to a path that may contain non-ASCII unicode characters."""
    if isinstance(path, Path):
        path = str(path)
    ext = os.path.splitext(path)[1]
    success, nparr = cv2.imencode(ext, img, params)
    if success:
        with open(path, 'wb') as f:
            f.write(nparr.tobytes())
        return True
    return False

# ---------------------------------------------------------
# 1. Coin Extraction & Masking
# ---------------------------------------------------------
def extract_coin(img_bgr_or_path):
    """
    Extracts coin foreground mask and bounding box from an image.
    Works robustly across white, colored, dark, or textured backgrounds.
    Accepts np.ndarray BGR or file path.
    Returns:
        mask (np.ndarray): 2D uint8 mask (0 or 255)
        bbox (tuple): (x, y, w, h)
        is_success (bool): Whether a clear coin contour was found
    """
    if isinstance(img_bgr_or_path, (str, Path)):
        img_bgr = cv2_imread_unicode(img_bgr_or_path)
    else:
        img_bgr = img_bgr_or_path

    if img_bgr is None:
        raise ValueError(f"Image could not be loaded: {img_bgr_or_path}")
        
    h, w = img_bgr.shape[:2]
    
    # Sample corner patches (5x5 pixels)
    c1 = img_bgr[:5, :5].reshape(-1, 3)
    c2 = img_bgr[:5, -5:].reshape(-1, 3)
    c3 = img_bgr[-5:, :5].reshape(-1, 3)
    c4 = img_bgr[-5:, -5:].reshape(-1, 3)
    corners = np.vstack([c1, c2, c3, c4])
    bg_mean = np.mean(corners, axis=0)
    bg_std = np.std(corners, axis=0)
    
    # Distance from background color
    diff = np.linalg.norm(img_bgr.astype(np.float32) - bg_mean, axis=-1)
    
    # Uniform corner threshold
    if np.mean(bg_std) < 30:
        thresh_val = max(25.0, np.mean(bg_std) * 2.5)
        mask = (diff > thresh_val).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        corner_density = (mask[:10, :10].mean() + mask[:10, -10:].mean() + 
                          mask[-10:, :10].mean() + mask[-10:, -10:].mean()) / 4.0
        if corner_density > 128:
            mask = cv2.bitwise_not(mask)
            
    # Morphological clean up
    k_size = max(5, int(min(h, w) * 0.02))
    if k_size % 2 == 0:
        k_size += 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    
    # Find best circular/coin contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_center = np.array([w / 2.0, h / 2.0])
    
    best_cnt = None
    best_score = -1e9
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.02 * h * w:
            continue
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            dist = np.linalg.norm(np.array([cx, cy]) - img_center)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
            score = area - (dist * min(h, w) * 2.0) + (circularity * area * 0.5)
            if score > best_score:
                best_score = score
                best_cnt = cnt
                
    final_mask = np.zeros((h, w), dtype=np.uint8)
    if best_cnt is not None:
        cv2.drawContours(final_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        x, y, cw, ch = cv2.boundingRect(best_cnt)
        success = True
    else:
        r = int(min(h, w) * 0.42)
        cx, cy = w // 2, h // 2
        cv2.circle(final_mask, (cx, cy), r, 255, -1)
        x, y, cw, ch = cx - r, cy - r, 2 * r, 2 * r
        success = False
        
    return final_mask, (x, y, cw, ch), success

# ---------------------------------------------------------
# 2. Background Generators
# ---------------------------------------------------------
def generate_random_background(target_size=(256, 256), rng=None):
    if rng is None:
        rng = np.random.default_rng()
        
    h, w = target_size
    bg_type = rng.choice(['wood', 'fabric', 'desk', 'gradient', 'noise', 'paper'])
    
    if bg_type == 'wood':
        base_color = rng.integers([40, 70, 110], [90, 140, 190], size=3)
        bg = np.full((h, w, 3), base_color, dtype=np.uint8)
        lines = rng.integers(15, 40)
        for _ in range(lines):
            y = rng.integers(0, h)
            thickness = rng.integers(1, 4)
            color_shift = rng.integers(-25, 25)
            c = np.clip(base_color + color_shift, 0, 255).astype(np.uint8).tolist()
            cv2.line(bg, (0, y), (w, y), c, thickness)
        bg = cv2.GaussianBlur(bg, (9, 3), 0)
        
    elif bg_type == 'fabric':
        base_color = rng.integers([30, 30, 30], [210, 210, 210], size=3)
        pattern = (rng.standard_normal((h, w)) * 12).astype(np.int16)
        bg = np.clip(base_color + pattern[:, :, None], 0, 255).astype(np.uint8)
        bg = cv2.GaussianBlur(bg, (3, 3), 0)
        
    elif bg_type == 'desk':
        base_color = rng.integers([160, 160, 160], [240, 240, 240], size=3)
        noise = rng.integers(-15, 15, size=(h, w, 3))
        bg = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        
    elif bg_type == 'gradient':
        c1 = rng.integers(50, 220, size=3)
        c2 = rng.integers(50, 220, size=3)
        grad = np.linspace(0, 1, h)[:, None]
        bg = (c1 * (1 - grad) + c2 * grad)[:, None, :] * np.ones((h, w, 3))
        bg = np.clip(bg, 0, 255).astype(np.uint8)
        
    elif bg_type == 'paper':
        base_color = rng.integers([220, 225, 230], [250, 250, 255], size=3)
        noise = rng.integers(-8, 8, size=(h, w, 3))
        bg = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        
    else:
        base = rng.integers(50, 200, size=3)
        noise = rng.integers(-30, 30, size=(h, w, 3))
        bg = np.clip(base + noise, 0, 255).astype(np.uint8)
        bg = cv2.GaussianBlur(bg, (5, 5), 0)
        
    return bg

# ---------------------------------------------------------
# 3. Synthetic Augmentation Pipeline
# ---------------------------------------------------------
def augment_coin_synthetic(
    img_bgr, 
    mask=None, 
    target_size=(224, 224), 
    intensity='medium', 
    rng=None,
    mix_raw_crop_prob=0.25
):
    """
    Generates realistic coin images with physical perturbations.
    Key fixes applied:
    - Scale is invariant to rotation angle (calculated before rotation).
    - Symmetric perspective tilt without directional bias.
    - Finger occlusion strictly at outer edges (<=15% area), never center.
    - Real crop mixing (25% prob) to reduce synthetic domain gap.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    h_in, w_in = img_bgr.shape[:2]
    if mask is None:
        mask, bbox, _ = extract_coin(img_bgr)
    else:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            bbox = cv2.boundingRect(largest)
        else:
            bbox = (0, 0, w_in, h_in)
            
    x, y, bw, bh = bbox
    th, tw = target_size

    # --- Mode A: Real Crop Mixing (25% prob) ---
    if rng.random() < mix_raw_crop_prob:
        pad_ratio = rng.uniform(0.08, 0.25)
        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)
        rx1 = max(0, x - pad_x)
        ry1 = max(0, y - pad_y)
        rx2 = min(w_in, x + bw + pad_x)
        ry2 = min(h_in, y + bh + pad_y)
        
        real_crop = img_bgr[ry1:ry2, rx1:rx2].copy()
        
        # Apply 0-360 rotation on real crop
        ang = rng.uniform(0, 360)
        rch, rcw = real_crop.shape[:2]
        rc_center = (rcw / 2.0, rch / 2.0)
        rmat = cv2.getRotationMatrix2D(rc_center, ang, 1.0)
        rc_rot = cv2.warpAffine(real_crop, rmat, (rcw, rch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Scale to canvas
        sc = rng.uniform(0.70, 0.95)
        cdim = int(min(th, tw) * sc)
        ar = rcw / max(1, rch)
        if ar >= 1.0:
            fw = cdim
            fh = max(1, int(cdim / ar))
        else:
            fh = cdim
            fw = max(1, int(cdim * ar))
            
        real_res = cv2.resize(rc_rot, (fw, fh), interpolation=cv2.INTER_AREA if sc < 1.0 else cv2.INTER_LINEAR)
        canvas = generate_random_background(target_size, rng=rng)
        
        mox = max(0, tw - fw)
        moy = max(0, th - fh)
        ox = rng.integers(0, mox + 1) if mox > 0 else 0
        oy = rng.integers(0, moy + 1) if moy > 0 else 0
        canvas[oy:oy+fh, ox:ox+fw] = real_res
        
        # Light lighting & noise
        b_shift = rng.integers(-20, 20)
        c_mult = rng.uniform(0.88, 1.12)
        composite = np.clip(canvas.astype(np.float32) * c_mult + b_shift, 0, 255).astype(np.uint8)
        if rng.random() < 0.35:
            composite = cv2.GaussianBlur(composite, (3, 3), 0)
        return composite

    # --- Mode B: Full Synthetic Composition ---
    pad = int(max(bw, bh) * 0.05)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_in, x + bw + pad)
    y2 = min(h_in, y + bh + pad)
    
    coin_crop = img_bgr[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2].copy()
    
    # 1. Invariant Scale Calculation (BEFORE rotation)
    orig_diam = max(bw, bh)
    scale_range = (0.65, 0.90) if intensity != 'heavy' else (0.50, 0.95)
    scale = rng.uniform(*scale_range)
    target_diam = max(24, int(min(th, tw) * scale))
    scale_factor = target_diam / max(1, orig_diam)
    
    coin_scaled = cv2.resize(coin_crop, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_LINEAR)
    mask_scaled = cv2.resize(mask_crop, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
    
    # 2. 0-360 Rotation (Strictly NO flip)
    angle = rng.uniform(0, 360)
    sch, scw = coin_scaled.shape[:2]
    center = (scw / 2.0, sch / 2.0)
    
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])
    new_w = int((sch * sin) + (scw * cos))
    new_h = int((sch * cos) + (scw * sin))
    rot_mat[0, 2] += (new_w / 2.0) - center[0]
    rot_mat[1, 2] += (new_h / 2.0) - center[1]
    
    coin_rot = cv2.warpAffine(coin_scaled, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask_rot = cv2.warpAffine(mask_scaled, rot_mat, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # 3. Symmetric Perspective Tilt
    if intensity in ['medium', 'heavy'] and rng.random() < 0.50:
        tilt = rng.uniform(0.04, 0.12 if intensity == 'medium' else 0.18)
        pts1 = np.float32([[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]])
        if rng.random() < 0.5:
            dx = int(new_w * tilt)
            pts2 = np.float32([[dx, 0], [new_w - dx, 0], [new_w, new_h], [0, new_h]])
        else:
            dy = int(new_h * tilt)
            pts2 = np.float32([[0, dy], [new_w, 0], [new_w, new_h - dy], [0, new_h]])
        persp_mat = cv2.getPerspectiveTransform(pts1, pts2)
        coin_rot = cv2.warpPerspective(coin_rot, persp_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask_rot = cv2.warpPerspective(mask_rot, persp_mat, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
    # Fit into canvas bounds
    fit_w = min(tw, new_w)
    fit_h = min(th, new_h)
    if fit_w != new_w or fit_h != new_h:
        coin_rot = cv2.resize(coin_rot, (fit_w, fit_h), interpolation=cv2.INTER_AREA)
        mask_rot = cv2.resize(mask_rot, (fit_w, fit_h), interpolation=cv2.INTER_NEAREST)
        
    mask_feathered = cv2.GaussianBlur(mask_rot.astype(np.float32) / 255.0, (5, 5), 0)
    
    # 4. Background placement
    bg = generate_random_background(target_size, rng=rng)
    max_ox = max(0, tw - fit_w)
    max_oy = max(0, th - fit_h)
    ox = rng.integers(0, max_ox + 1) if max_ox > 0 else 0
    oy = rng.integers(0, max_oy + 1) if max_oy > 0 else 0
    
    bg_roi = bg[oy:oy+fit_h, ox:ox+fit_w].astype(np.float32)
    coin_float = coin_rot.astype(np.float32)
    alpha = mask_feathered[:, :, None]
    blended_roi = coin_float * alpha + bg_roi * (1.0 - alpha)
    bg[oy:oy+fit_h, ox:ox+fit_w] = np.clip(blended_roi, 0, 255).astype(np.uint8)
    composite = bg.copy()
    
    # 5. Glare
    if rng.random() < 0.45:
        glare_cx = int(ox + fit_w * rng.uniform(0.25, 0.75))
        glare_cy = int(oy + fit_h * rng.uniform(0.25, 0.75))
        glare_radius = int(min(fit_w, fit_h) * rng.uniform(0.12, 0.28))
        glare_mask = np.zeros((th, tw), dtype=np.float32)
        cv2.circle(glare_mask, (glare_cx, glare_cy), glare_radius, 1.0, -1)
        glare_mask = cv2.GaussianBlur(glare_mask, (glare_radius*2+1, glare_radius*2+1), 0)
        glare_intensity = rng.uniform(25, 70)
        composite = np.clip(composite.astype(np.float32) + (glare_mask[:, :, None] * glare_intensity), 0, 255).astype(np.uint8)
        
    # 6. Simulated Finger Occlusion (Strictly Outer Boundaries, <=15% coin area, NEVER center)
    if rng.random() < 0.35:
        finger_color = rng.integers([110, 130, 170], [150, 180, 225], size=3)
        # Position only at extreme edges (0.08 or 0.92)
        edge_x = rng.choice([0.08, 0.92])
        edge_y = rng.choice([0.08, 0.92])
        fcx = int(ox + fit_w * edge_x)
        fcy = int(oy + fit_h * edge_y)
        # Bounded semi-axes so area <= 15%
        axes = (int(fit_w * rng.uniform(0.10, 0.18)), int(fit_h * rng.uniform(0.08, 0.14)))
        f_angle = float(rng.uniform(0, 180))
        f_mask = np.zeros((th, tw), dtype=np.float32)
        cv2.ellipse(f_mask, (fcx, fcy), axes, f_angle, 0, 360, 1.0, -1)
        f_mask = cv2.GaussianBlur(f_mask, (11, 11), 0)
        composite = (composite.astype(np.float32) * (1.0 - f_mask[:, :, None]) + finger_color * f_mask[:, :, None]).clip(0, 255).astype(np.uint8)
        
    # 7. Lighting / Brightness & Contrast
    b_shift = rng.integers(-25, 25)
    c_mult = rng.uniform(0.85, 1.15)
    composite = np.clip(composite.astype(np.float32) * c_mult + b_shift, 0, 255).astype(np.uint8)
    
    # 8. Motion / Gaussian blur
    if rng.random() < 0.35:
        b_k = rng.choice([3, 5])
        composite = cv2.GaussianBlur(composite, (b_k, b_k), 0)
        
    # 9. Noise & JPEG Compression
    if rng.random() < 0.40:
        noise = rng.normal(0, rng.uniform(2, 8), composite.shape)
        composite = np.clip(composite.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    if rng.random() < 0.45:
        quality = rng.integers(65, 95)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', composite, encode_param)
        composite = cv2.imdecode(encimg, 1)
        
    return composite

# ---------------------------------------------------------
# 4. Distractor / "Bukan Koin" Generator
# ---------------------------------------------------------
def generate_distractor_sample(target_size=(224, 224), rng=None):
    if rng is None:
        rng = np.random.default_rng()
        
    h, w = target_size
    mode = rng.choice(['pure_bg', 'fingers', 'round_distractor'])
    
    if mode == 'pure_bg':
        img = generate_random_background(target_size, rng=rng)
        
    elif mode == 'fingers':
        img = generate_random_background(target_size, rng=rng)
        flesh = rng.integers([110, 130, 170], [150, 180, 230], size=3)
        for _ in range(rng.integers(2, 4)):
            cx = int(rng.integers(0, w))
            cy = int(rng.integers(0, h))
            axes = (int(rng.integers(int(w*0.2), int(w*0.5))), int(rng.integers(int(h*0.1), int(h*0.25))))
            angle = float(rng.uniform(0, 180))
            cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, [int(c) for c in flesh], -1)
        img = cv2.GaussianBlur(img, (15, 15), 0)
        
    else:
        img = generate_random_background(target_size, rng=rng)
        cx = int(w // 2 + rng.integers(-30, 30))
        cy = int(h // 2 + rng.integers(-30, 30))
        radius = int(min(w, h) * rng.uniform(0.25, 0.40))
        color_choices = [
            (20, 20, 210),
            (210, 50, 20),
            (20, 190, 30),
            (30, 30, 30)
        ]
        cap_color = color_choices[int(rng.integers(0, len(color_choices)))]
        cv2.circle(img, (cx, cy), radius, cap_color, -1)
        if rng.random() < 0.5:
            cv2.circle(img, (cx, cy), int(radius * 0.4), (180, 180, 180), -1)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
    return img

# ---------------------------------------------------------
# 5. Live Sequence Generator (PyDataset for Keras 3 / TF 2.16)
# ---------------------------------------------------------
try:
    import tensorflow as tf
    
    class LiveCoinDataset(tf.keras.utils.PyDataset):
        """
        Dynamic on-the-fly augmentation dataset generator for Keras.
        Generates fresh synthetic & real mixed augmentations for every epoch.
        Guarantees that models receive diverse gradient updates throughout training.
        """
        def __init__(
            self, 
            manifest_items, 
            label_map, 
            batch_size=16, 
            target_size=(224, 224), 
            steps_per_epoch=25, 
            intensity='medium', 
            preprocess_fn=None,
            is_training=True,
            seed=42,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.items = manifest_items # list of (img_bgr, mask, label_str)
            self.label_map = label_map
            self.num_classes = len(label_map)
            self.batch_size = batch_size
            self.target_size = target_size
            self.steps_per_epoch = steps_per_epoch
            self.intensity = intensity
            self.preprocess_fn = preprocess_fn
            self.is_training = is_training
            self.rng = np.random.default_rng(seed)

        def __len__(self):
            return self.steps_per_epoch

        def __getitem__(self, idx):
            batch_x = []
            batch_y = []
            
            for _ in range(self.batch_size):
                # Sample random item from items
                item_idx = self.rng.integers(0, len(self.items))
                img_bgr, mask, label_str = self.items[item_idx]
                
                if self.is_training:
                    aug_bgr = augment_coin_synthetic(
                        img_bgr, 
                        mask=mask, 
                        target_size=self.target_size, 
                        intensity=self.intensity, 
                        rng=self.rng,
                        mix_raw_crop_prob=0.25
                    )
                else:
                    # Deterministic/clean crop for validation
                    th, tw = self.target_size
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
                        pad = int(max(w, h) * 0.08)
                        x1 = max(0, x - pad)
                        y1 = max(0, y - pad)
                        x2 = min(img_bgr.shape[1], x + w + pad)
                        y2 = min(img_bgr.shape[0], y + h + pad)
                        crop = img_bgr[y1:y2, x1:x2]
                    else:
                        crop = img_bgr
                    aug_bgr = cv2.resize(crop, (tw, th))
                    
                # BGR -> RGB
                img_rgb = cv2.cvtColor(aug_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
                if self.preprocess_fn is not None:
                    img_rgb = self.preprocess_fn(img_rgb)
                    
                batch_x.append(img_rgb)
                
                one_hot = np.zeros(self.num_classes, dtype=np.float32)
                one_hot[self.label_map[label_str]] = 1.0
                batch_y.append(one_hot)
                
            return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

except ImportError:
    pass

# ---------------------------------------------------------
# 6. Offline Batch Cache Generator (Fallback / Offline support)
# ---------------------------------------------------------
def generate_offline_augmented_dataset(
    manifest_df,
    output_dir,
    samples_per_class=200,
    target_size=(224, 224),
    intensity='medium',
    include_distractor=False,
    seed=42
):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    
    grouped = manifest_df.groupby('class')
    records = []
    
    for cls_name, group in grouped:
        cls_dir = os.path.join(output_dir, str(cls_name))
        os.makedirs(cls_dir, exist_ok=True)
        
        src_items = []
        for _, row in group.iterrows():
            img_bgr = cv2_imread_unicode(row['path'])
            mask, bbox, _ = extract_coin(img_bgr)
            src_items.append((row['file'], img_bgr, mask))
            
        n_sources = len(src_items)
        for i in range(samples_per_class):
            src_idx = i % n_sources
            src_file, img_bgr, mask = src_items[src_idx]
            
            aug_bgr = augment_coin_synthetic(
                img_bgr, 
                mask=mask, 
                target_size=target_size, 
                intensity=intensity, 
                rng=rng,
                mix_raw_crop_prob=0.25
            )
            out_filename = f"synth_{cls_name}_{i:04d}_src_{os.path.splitext(src_file)[0]}.png"
            out_path = os.path.join(cls_dir, out_filename)
            cv2_imwrite_unicode(out_path, aug_bgr)
            
            records.append({
                'class': str(cls_name),
                'file': out_filename,
                'path': out_path,
                'src_file': src_file,
                'is_synthetic': True
            })
            
    if include_distractor:
        cls_name = 'bukan_koin'
        cls_dir = os.path.join(output_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(samples_per_class):
            dist_bgr = generate_distractor_sample(target_size=target_size, rng=rng)
            out_filename = f"synth_{cls_name}_{i:04d}.png"
            out_path = os.path.join(cls_dir, out_filename)
            cv2_imwrite_unicode(out_path, dist_bgr)
            records.append({
                'class': cls_name,
                'file': out_filename,
                'path': out_path,
                'src_file': 'synthetic_distractor',
                'is_synthetic': True
            })
            
    import pandas as pd
    manifest_out = pd.DataFrame(records)
    manifest_out.to_csv(os.path.join(output_dir, 'manifest.csv'), index=False)
    return manifest_out
