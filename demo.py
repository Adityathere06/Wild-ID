import gradio as gr
import os
import sys
import shutil
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

sys.path.append('src')

from feature_extractor import SnowLeopardFeatureExtractor
from matcher import SnowLeopardMatcher
from tiger_detector import TigerDetector
from utils import (
    parse_reference_structure,
    save_features,
    load_features,
    validate_dataset
)

# Global variables
extractor = None
matcher = None
detector = None
reference_features = {}

THRESHOLD = 0.96
WEBCAM_THRESHOLD = 0.90
WEBCAM_MARGIN = 0.03  # min gap between #1 and #2 to accept a webcam match
PKL_PATH = 'models/reference_features.pkl'
REFERENCE_DIR = 'data/reference'

def initialize_system():
    global extractor, matcher, detector
    extractor = SnowLeopardFeatureExtractor(device='cpu')
    matcher = SnowLeopardMatcher()
    detector = TigerDetector(device='cpu')

# ── Database ─────────────────────────────────────────────────────────────────

def load_existing_database():
    """Load already-built .pkl into matcher — no rebuilding."""
    global extractor, matcher, reference_features

    if extractor is None:
        initialize_system()

    if not os.path.exists(PKL_PATH):
        return (
            f"No existing database found at:\n  {PKL_PATH}\n\n"
            "Use 'Build Database from Images' to create it first."
        )

    reference_features = load_features(PKL_PATH)

    if not reference_features:
        return "Database file exists but is empty. Rebuild it."

    # Reconstruct matcher from pkl
    matcher = SnowLeopardMatcher()
    for img_path, features in reference_features.items():
        parts = Path(img_path).parts
        try:
            name = parts[-2]  # folder name = tiger identity
        except IndexError:
            name = "unknown"
        matcher.add_reference(name, features, img_path)

    stats = matcher.get_statistics()
    result = (
        f"Existing database loaded.\n\n"
        f"Individuals : {stats['total_individuals']}\n"
        f"Total images: {stats['total_images']}\n\n"
        f"Known tigers:\n" +
        "\n".join(f"  {name}" for name in sorted(stats['individuals']))
    )
    return result


def build_reference_database():
    """Scan data/reference/, extract embeddings, save new pkl.
    Yields live status updates with elapsed + estimated remaining time."""
    global extractor, matcher, reference_features

    if extractor is None:
        initialize_system()

    matcher = SnowLeopardMatcher()
    reference_features = {}

    individuals = parse_reference_structure(REFERENCE_DIR)

    if not individuals:
        yield "No reference images found. Add images to data/reference/"
        return

    # Collect all (name, img_path) pairs upfront so we know total count
    all_tasks = []
    for name, image_paths in individuals.items():
        for img_path in image_paths:
            all_tasks.append((name, img_path))

    total = len(all_tasks)
    build_start = time.time()

    def fmt_time(seconds):
        """Format seconds into Xm Ys or Xs."""
        seconds = max(0, int(seconds))
        if seconds >= 60:
            m, s = divmod(seconds, 60)
            return f"{m}m {s:02d}s"
        return f"{seconds}s"

    for idx, (name, img_path) in enumerate(all_tasks):
        elapsed = time.time() - build_start
        done = idx  # images completed so far

        # Estimate remaining time
        if done > 0:
            avg_per_image = elapsed / done
            remaining_est = avg_per_image * (total - done)
            remaining_str = fmt_time(remaining_est)
        else:
            remaining_str = "calculating..."

        # Live status shown while processing this image
        live_status = (
            f"BUILDING DATABASE...\n"
            f"{'─'*40}\n\n"
            f"Processing : {os.path.basename(img_path)}\n"
            f"Progress   : {done}/{total} images done\n\n"
            f"Elapsed    : {fmt_time(elapsed)}\n"
            f"Remaining  : {remaining_str}"
        )
        yield live_status

        features = extractor.extract_features(img_path)
        if features is not None:
            matcher.add_reference(name, features, img_path)
            reference_features[img_path] = features

    save_features(reference_features, PKL_PATH)

    total_elapsed = time.time() - build_start
    stats = matcher.get_statistics()

    result = (
        f"Database built and saved.\n"
        f"{'─'*40}\n\n"
        f"Individuals loaded : {stats['total_individuals']}\n"
        f"Total images       : {stats['total_images']}\n"
        f"Total time taken   : {fmt_time(total_elapsed)}\n\n"
        f"Known tigers:\n" +
        "\n".join(f"  {name}" for name in sorted(stats['individuals']))
    )
    yield result


# ── Identification ────────────────────────────────────────────────────────────

last_query_path = None

def identify_leopard(image):
    global extractor, matcher, detector, last_query_path

    if extractor is None or detector is None:
        initialize_system()

    if matcher is None or not matcher.reference_features:
        return "Load or build the reference database first.", None, []

    if image is None:
        return "Please upload an image to identify.", None, []

    try:
        # 1. Species check: Verify if the image contains a tiger
        is_tiger, species_label, tiger_conf, top_pred, top_prob = detector.predict(image)

        if not is_tiger:
            result = (
                f"RESULT — NOT A TIGER DETECTED\n"
                f"{'─'*40}\n\n"
                f"The submitted image does not contain a tiger.\n\n"
                f"Detected subject : {species_label}\n"
                f"Confidence       : {top_prob:.1%}\n\n"
                f"This system is strictly trained for Tiger Identification.\n"
                f"Please provide an image of a tiger."
            )
            return result, image, []

        temp_path = 'temp_query.jpg'
        image.save(temp_path)
        last_query_path = temp_path

        features = extractor.extract_features(temp_path)

        if features is None:
            return "Feature extraction failed.", None, []

        best_match, confidence, all_scores = matcher.find_match(features, threshold=THRESHOLD)
        top_matches = matcher.get_top_matches(features, top_k=3)

        reference_images = []
        if best_match != "Unknown":
            ref_paths = matcher.get_reference_images(best_match)
            reference_images = [p for p in ref_paths if os.path.exists(p)]

        if best_match == "Unknown":
            result = (
                f"RESULT — NEW / UNKNOWN TIGER DETECTED\n"
                f"{'─'*40}\n\n"
                f"Tiger confirmed, but no matching individual found in the reference database.\n"
                f"This appears to be a new individual.\n\n"
                f"Confidence threshold : {THRESHOLD:.0%}\n"
                f"Best similarity score: {confidence:.2%}\n\n"
                f"Top similarities:\n"
            )
        else:
            result = (
                f"RESULT — MATCH FOUND\n"
                f"{'─'*40}\n\n"
                f"Identity   : {best_match}\n"
                f"Confidence : {confidence:.2%}\n\n"
                f"Top similarities:\n"
            )

        for i, (name, score) in enumerate(top_matches, 1):
            bar = '█' * int(score * 20)
            result += f"\n  {i}. {name}  {score:.2%}  {bar}"

        if reference_images:
            result += f"\n\n{len(reference_images)} reference image(s) shown below."

        return result, image, reference_images

    except Exception as e:
        return f"Error: {str(e)}", None, []



def preprocess_webcam_frame(pil_image):
    """Preprocess a webcam frame to improve matching quality.

    When a phone screen is shown to a webcam, the captured image suffers from:
      - Screen glare / reflections
      - Moiré patterns from the screen pixel grid
      - Lower contrast due to ambient lighting
      - Slight blur from camera autofocus

    This function counteracts those artifacts.
    """
    image = pil_image.convert('RGB')

    # 1. Boost contrast to counteract washed-out screen captures
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.3)

    # 2. Boost colour saturation (phone screens can wash out colours)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.2)

    # 3. Slight sharpening to recover detail lost through the screen
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)

    # 4. Mild median filter to reduce moiré noise without losing detail
    image = image.filter(ImageFilter.MedianFilter(size=3))

    # 5. One more sharpen pass after denoising to keep edges crisp
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.3)

    return image


def identify_from_webcam(frame):
    """Identify a tiger from a webcam frame (numpy array or PIL image)."""
    global extractor, matcher, detector

    if extractor is None or detector is None:
        initialize_system()

    if matcher is None or not matcher.reference_features:
        return "Load or build the reference database first.", None, []

    if frame is None:
        return "No frame captured. Click the camera button to take a photo.", None, []

    try:
        # Convert numpy array from Gradio webcam to PIL
        if isinstance(frame, np.ndarray):
            pil_image = Image.fromarray(frame)
        else:
            pil_image = frame

        # Preprocess to counteract phone-screen artifacts
        processed = preprocess_webcam_frame(pil_image)

        # 1. Species check: Verify if the camera view contains a tiger
        # Check both raw captured frame and preprocessed frame
        is_tiger, species_label, tiger_conf, top_pred, top_prob = detector.predict(pil_image)
        if not is_tiger:
            is_tiger_proc, species_label_proc, tiger_conf_proc, top_pred_proc, top_prob_proc = detector.predict(processed)
            if is_tiger_proc:
                is_tiger = True
                species_label = species_label_proc
                top_prob = top_prob_proc
            else:
                lbl = species_label if top_prob >= top_prob_proc else species_label_proc
                prob = max(top_prob, top_prob_proc)
                result = (
                    f"WEBCAM RESULT — NOT A TIGER DETECTED\n"
                    f"{'─'*40}\n\n"
                    f"No tiger was detected in the webcam view.\n\n"
                    f"Detected subject : {lbl}\n"
                    f"Confidence       : {prob:.1%}\n\n"
                    f"Please point the webcam or hold your phone showing an image of a tiger."
                )
                return result, processed, []

        # 2. Individual Identification
        # Multi-crop extraction for robustness
        features = extractor.extract_features_multi_crop(processed, num_crops=5)

        if features is None:
            # Fallback to single-shot extraction
            features = extractor.extract_features_from_pil(processed)

        if features is None:
            return "Feature extraction failed.", None, []

        # Use webcam-specific threshold
        best_match, confidence, all_scores = matcher.find_match(
            features, threshold=WEBCAM_THRESHOLD
        )
        top_matches = matcher.get_top_matches(features, top_k=5)

        # ── Margin check ──────────────────────────────────────────────
        # An out-of-dataset tiger will score similarly against ALL
        # individuals (generic "tiger-ness"). A genuine match will
        # have one individual clearly ahead of the rest.
        # If the gap between #1 and #2 is too small, reject the match.
        margin_rejected = False
        margin = 0.0
        if best_match != "Unknown" and len(top_matches) >= 2:
            top1_score = top_matches[0][1]
            top2_score = top_matches[1][1]
            margin = top1_score - top2_score
            if margin < WEBCAM_MARGIN:
                margin_rejected = True
                best_match = "Unknown"

        reference_images = []
        if best_match != "Unknown":
            ref_paths = matcher.get_reference_images(best_match)
            reference_images = [p for p in ref_paths if os.path.exists(p)]

        if best_match == "Unknown":
            reason = ""
            if margin_rejected:
                reason = (
                    f"Pattern margin between top-1 and top-2 ({margin:.4f}) "
                    f"is below {WEBCAM_MARGIN} — ambiguous pattern match.\n"
                )
            result = (
                f"WEBCAM RESULT — NEW / UNKNOWN TIGER DETECTED\n"
                f"{'─'*40}\n\n"
                f"Tiger confirmed, but no matching individual found in reference database.\n"
                f"{reason}"
                f"Webcam threshold     : {WEBCAM_THRESHOLD:.0%}\n"
                f"Best similarity score: {confidence:.2%}\n\n"
                f"Top similarities:\n"
            )
        else:
            result = (
                f"WEBCAM RESULT — MATCH FOUND\n"
                f"{'─'*40}\n\n"
                f"Identity   : {best_match}\n"
                f"Confidence : {confidence:.2%}\n\n"
                f"Top similarities:\n"
            )

        for i, (name, score) in enumerate(top_matches, 1):
            bar = '█' * int(score * 20)
            result += f"\n  {i}. {name}  {score:.2%}  {bar}"

        if reference_images:
            result += f"\n\n{len(reference_images)} reference image(s) shown below."

        return result, processed, reference_images

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", None, []


def add_new_tiger(tiger_name, image):
    """Add the query image as a new tiger into the reference database."""
    global extractor, matcher, detector, reference_features, last_query_path

    if extractor is None or detector is None:
        initialize_system()

    tiger_name = tiger_name.strip()
    if not tiger_name:
        return "Enter a tiger name first."

    if image is None and last_query_path is None:
        return "No query image available to add."

    if matcher is None:
        return "Database not loaded. Load or build the database first."

    # Validate that the image actually contains a tiger
    img_to_check = image
    if img_to_check is None and last_query_path and os.path.exists(last_query_path):
        try:
            img_to_check = Image.open(last_query_path)
        except Exception:
            pass

    if img_to_check is not None:
        is_tiger, species_label, _, _, _ = detector.predict(img_to_check)
        if not is_tiger:
            return f"Cannot add image: Detected subject is '{species_label}', not a tiger."

    # Save image into reference folder
    tiger_folder = os.path.join(REFERENCE_DIR, tiger_name)
    os.makedirs(tiger_folder, exist_ok=True)

    existing = [f for f in os.listdir(tiger_folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    new_idx = len(existing) + 1
    dest_path = os.path.join(tiger_folder, f"img_{new_idx:03d}.jpg")

    if image is not None:
        image.save(dest_path)
    elif last_query_path and os.path.exists(last_query_path):
        shutil.copy(last_query_path, dest_path)
    else:
        return "Could not retrieve query image."

    # Extract and register
    features = extractor.extract_features(dest_path)
    if features is None:
        return "Feature extraction failed. Image not added."

    matcher.add_reference(tiger_name, features, dest_path)
    reference_features[dest_path] = features

    # Persist
    save_features(reference_features, PKL_PATH)

    stats = matcher.get_statistics()
    return (
        f"Added successfully.\n\n"
        f"Tiger name   : {tiger_name}\n"
        f"Saved to     : {dest_path}\n\n"
        f"Database now : {stats['total_individuals']} individuals, "
        f"{stats['total_images']} images total."
    )


def delete_tiger(tiger_name):
    """Remove a tiger completely from live matcher, pkl, and reference folder."""
    global matcher, reference_features

    tiger_name = tiger_name.strip()
    if not tiger_name:
        return "Enter a tiger name to delete."

    if matcher is None:
        return "Database not loaded. Load or build the database first."

    if tiger_name not in matcher.reference_features:
        known = sorted(matcher.reference_features.keys())
        return (
            f"'{tiger_name}' not found in database.\n\n"
            f"Known tigers:\n" + "\n".join(f"  {n}" for n in known)
        )

    # Remove from matcher
    del matcher.reference_features[tiger_name]
    del matcher.reference_names[tiger_name]

    # Remove from pkl dict
    keys_to_remove = [k for k in reference_features if Path(k).parts[-2] == tiger_name]
    for k in keys_to_remove:
        del reference_features[k]

    # Save updated pkl
    save_features(reference_features, PKL_PATH)

    # Remove folder from disk
    tiger_folder = os.path.join(REFERENCE_DIR, tiger_name)
    folder_removed = False
    if os.path.exists(tiger_folder):
        shutil.rmtree(tiger_folder)
        folder_removed = True

    stats = matcher.get_statistics()
    return (
        f"Deleted.\n\n"
        f"Tiger removed    : {tiger_name}\n"
        f"Folder deleted   : {'yes — ' + tiger_folder if folder_removed else 'not found on disk'}\n"
        f"Images removed   : {len(keys_to_remove)}\n\n"
        f"Database now : {stats['total_individuals']} individuals, "
        f"{stats['total_images']} images total."
    )


# ── Batch ─────────────────────────────────────────────────────────────────────

def batch_identify(test_dir='data/test'):
    global extractor, matcher, detector

    if extractor is None or detector is None:
        initialize_system()

    if matcher is None or not matcher.reference_features:
        return "Load or build the reference database first."

    from utils import load_images_from_directory
    test_images = load_images_from_directory(test_dir)

    if not test_images:
        return f"No test images found in {test_dir}"

    results = f"BATCH RESULTS — {len(test_images)} images\n{'─'*50}\n\n"

    for img_path in test_images:
        # Check if image contains a tiger
        try:
            pil_img = Image.open(img_path).convert('RGB')
            is_tiger, species_label, _, _, prob = detector.predict(pil_img)
            if not is_tiger:
                results += f"{os.path.basename(img_path)}\n"
                results += f"  → NOT A TIGER ({species_label})  [Skipped]\n\n"
                continue
        except Exception:
            pass

        features = extractor.extract_features(img_path)
        if features is not None:
            best_match, confidence, _ = matcher.find_match(features, threshold=THRESHOLD)
            status = "NEW TIGER" if best_match == "Unknown" else best_match
            results += f"{os.path.basename(img_path)}\n"
            results += f"  → {status}  ({confidence:.2%})\n\n"

    return results


# ── CSS ───────────────────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Reset & Base ────────────────────────────────────── */
footer, .built-with, #svelte-announcer,
.gradio-container > .footer { display: none !important; }

:root {
    --bg-primary: #09090b;
    --bg-secondary: #111113;
    --bg-card: #16161a;
    --bg-elevated: #1c1c21;
    --border-subtle: #27272a;
    --border-default: #3f3f46;
    --text-primary: #fafafa;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;
    --accent: #f59e0b;
    --accent-hover: #fbbf24;
    --accent-subtle: rgba(245, 158, 11, 0.08);
    --accent-glow: rgba(245, 158, 11, 0.15);
    --danger: #ef4444;
    --success: #22c55e;
    --radius: 12px;
    --radius-sm: 8px;
}

body, .gradio-container {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Animated Header ─────────────────────────────────── */
.hero-header {
    position: relative;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 0.5rem;
    text-align: center;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(135deg,
        rgba(245, 158, 11, 0.06) 0%,
        rgba(245, 158, 11, 0.02) 30%,
        transparent 60%,
        rgba(245, 158, 11, 0.04) 100%);
    animation: headerShimmer 8s ease-in-out infinite;
}

@keyframes headerShimmer {
    0%, 100% { opacity: 0.7; }
    50% { opacity: 1; }
}

.hero-header .logo-icon {
    font-size: 2.8rem;
    display: inline-block;
    animation: tigerPulse 3s ease-in-out infinite;
    filter: drop-shadow(0 0 20px rgba(245, 158, 11, 0.3));
}

@keyframes tigerPulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.08); }
}

.hero-header h1 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    color: var(--text-primary) !important;
    margin: 0.6rem 0 0.3rem 0 !important;
    text-transform: uppercase;
}

.hero-header .subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: var(--text-muted);
    letter-spacing: 0.04em;
    font-weight: 400;
}

.hero-header .accent-line {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), transparent);
    margin: 1rem auto 0 auto;
    border-radius: 2px;
}

/* ── Tab Navigation ──────────────────────────────────── */
.tab-nav {
    background: var(--bg-secondary) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) var(--radius) 0 0 !important;
    padding: 0 0.5rem !important;
    gap: 0 !important;
}

.tab-nav button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.85rem 1.3rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
}

.tab-nav button:hover {
    color: var(--text-secondary) !important;
    background: var(--accent-subtle) !important;
}

.tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: var(--accent-subtle) !important;
}

/* ── Primary Button ──────────────────────────────────── */
button.primary {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    background: linear-gradient(135deg, var(--accent), #d97706) !important;
    color: #000 !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.75rem 1.8rem !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 0 20px rgba(245, 158, 11, 0.15),
                0 4px 12px rgba(0, 0, 0, 0.3) !important;
    position: relative !important;
    overflow: hidden !important;
}

button.primary:hover {
    background: linear-gradient(135deg, var(--accent-hover), #e6a117) !important;
    box-shadow: 0 0 30px rgba(245, 158, 11, 0.25),
                0 6px 20px rgba(0, 0, 0, 0.4) !important;
    transform: translateY(-1px) !important;
}

button.primary:active {
    transform: translateY(0) !important;
}

/* ── Secondary Button ────────────────────────────────── */
button.secondary {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    background: var(--bg-elevated) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.7rem 1.5rem !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

button.secondary:hover {
    background: var(--bg-card) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    box-shadow: 0 0 15px rgba(245, 158, 11, 0.08) !important;
}

/* ── Text Areas & Inputs ─────────────────────────────── */
textarea, .gr-textbox textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    background: var(--bg-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    line-height: 1.8 !important;
    padding: 1rem !important;
    transition: border-color 0.3s !important;
}

textarea:focus, .gr-textbox textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-subtle) !important;
    outline: none !important;
}

input[type="text"], .gr-textbox input {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    background: var(--bg-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    padding: 0.6rem 0.9rem !important;
    transition: border-color 0.3s !important;
}

input[type="text"]:focus, .gr-textbox input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-subtle) !important;
    outline: none !important;
}

/* ── Labels ──────────────────────────────────────────── */
label span, .gr-form label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

/* ── Image Panels ────────────────────────────────────── */
.gr-image, .gr-gallery {
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
    background: var(--bg-secondary) !important;
    overflow: hidden !important;
    transition: border-color 0.3s !important;
}

.gr-image:hover, .gr-gallery:hover {
    border-color: var(--border-default) !important;
}

/* ── Section Panels ──────────────────────────────────── */
.section-panel {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin: 0.8rem 0;
    transition: border-color 0.3s, box-shadow 0.3s;
}

.section-panel:hover {
    border-color: var(--border-default);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
}

/* ── Info Note ───────────────────────────────────────── */
.info-note {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 0.04em;
    background: var(--accent-subtle);
    border-left: 3px solid var(--accent);
    padding: 0.6rem 1rem;
    margin: 0.5rem 0 1.2rem 0;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
}

/* ── Divider ─────────────────────────────────────────── */
.divider {
    border: none;
    border-top: 1px solid var(--border-subtle);
    margin: 1.2rem 0;
}

/* ── Markdown ────────────────────────────────────────── */
.gr-markdown h2, .gr-markdown h3, .gr-markdown h4 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    padding-bottom: 0.5rem !important;
    margin-top: 1.2rem !important;
}

.gr-markdown p, .gr-markdown li {
    font-size: 0.84rem !important;
    color: var(--text-secondary) !important;
    line-height: 1.8 !important;
}

.gr-markdown code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.74rem !important;
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--accent) !important;
    border-radius: 4px !important;
    padding: 0.15rem 0.4rem !important;
}

.gr-markdown pre {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.74rem !important;
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-sm) !important;
    padding: 1rem !important;
}

/* ── Stat Cards (Info Tab) ───────────────────────────── */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stat-card:hover {
    border-color: var(--accent);
    box-shadow: 0 0 20px var(--accent-glow);
    transform: translateY(-2px);
}

.stat-card .stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    display: block;
    margin-bottom: 0.3rem;
}

.stat-card .stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Pipeline Badge ──────────────────────────────────── */
.pipeline-step {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    padding: 0.35rem 0.9rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-secondary);
    margin: 0.2rem;
}

.pipeline-step .step-icon {
    font-size: 0.85rem;
}

/* ── Status indicator ────────────────────────────────── */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
    margin-right: 6px;
    animation: statusPulse 2s ease-in-out infinite;
}

@keyframes statusPulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(245, 158, 11, 0); }
}

/* ── Delete zone ─────────────────────────────────────── */
.danger-zone {
    background: rgba(239, 68, 68, 0.04);
    border: 1px solid rgba(239, 68, 68, 0.15);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}

.danger-zone-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--danger);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb {
    background: var(--border-default);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Smooth global transitions ───────────────────────── */
*, *::before, *::after {
    transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
}
"""

# ── Interface ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="WildID \u2014 Tiger Identification") as demo:

    # ── Hero Header ──────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="hero-header">
        <div class="logo-icon">\U0001F42F</div>
        <h1>WildID</h1>
        <div class="subtitle">AI-Powered Individual Tiger Identification System</div>
        <div class="accent-line"></div>
    </div>
    """)

    # ── Setup ────────────────────────────────────────────────────────────────
    with gr.Tab("\u2699\ufe0f Setup"):
        gr.HTML("""
        <div class="info-note">
            <span class="status-dot"></span>
            Load an existing reference database or build a new one from images in <code>data/reference/</code>
        </div>
        """)

        gr.Markdown("#### Reference Database")

        with gr.Row():
            load_btn  = gr.Button("\u26a1 Load Existing Database",     variant="primary")
            build_btn = gr.Button("\U0001F527 Build Database from Images", variant="secondary")

        setup_output = gr.Textbox(label="Status", lines=14)

        load_btn.click(fn=load_existing_database, outputs=setup_output)
        build_btn.click(fn=build_reference_database, outputs=setup_output)

        gr.HTML('<hr class="divider">')

        gr.HTML("""
        <div class="danger-zone">
            <div class="danger-zone-label">\u26a0 Danger Zone</div>
            <p style="font-size: 0.78rem; color: #a1a1aa; margin: 0;">
                Permanently remove a tiger from the live database, .pkl file, and reference folder.
            </p>
        </div>
        """)
        with gr.Row():
            delete_name_input = gr.Textbox(
                label="Tiger Name / ID to Delete",
                placeholder="e.g.  tiger_030",
                scale=3
            )
            delete_btn = gr.Button("\U0001F5D1 Delete Tiger", variant="secondary", scale=1)
        delete_output = gr.Textbox(label="", lines=5, show_label=False)

        delete_btn.click(
            fn=delete_tiger,
            inputs=delete_name_input,
            outputs=delete_output
        )

    # ── Identify ─────────────────────────────────────────────────────────────
    with gr.Tab("\U0001F50D Identify"):
        gr.HTML(f"""
        <div class="info-note">
            <span class="status-dot"></span>
            Upload a tiger image for individual identification &nbsp;\u00b7&nbsp;
            Similarity threshold: <strong>{THRESHOLD:.0%}</strong> &nbsp;\u00b7&nbsp;
            Two-stage pipeline: Species Check \u2192 Pattern Match
        </div>
        """)

        gr.HTML("""
        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem;">
            <span class="pipeline-step"><span class="step-icon">\U0001F9EC</span> MobileNetV3 Species Filter</span>
            <span class="pipeline-step"><span class="step-icon">\U0001F9E0</span> ViT-Base Embedding</span>
            <span class="pipeline-step"><span class="step-icon">\U0001F3AF</span> Cosine Similarity</span>
        </div>
        """)

        gr.Markdown("#### Upload & Identify")
        with gr.Row():
            with gr.Column(scale=1):
                input_image  = gr.Image(label="Query Image", type="pil")
                identify_btn = gr.Button("\U0001F50D Identify Tiger", variant="primary")
            with gr.Column(scale=1):
                output_text  = gr.Textbox(label="Identification Result", lines=14)
                output_image = gr.Image(visible=False)

        gr.HTML('<hr class="divider">')

        gr.HTML("""
        <div class="info-note" style="border-left-color: #22c55e; background: rgba(34, 197, 94, 0.05);">
            New tiger detected? Name it and add it to the reference database below.
        </div>
        """)
        with gr.Row():
            tiger_name_input = gr.Textbox(
                label="Tiger Name / ID",
                placeholder="e.g.  tiger_275",
                scale=3
            )
            add_btn = gr.Button("\u2795 Add to Database", variant="secondary", scale=1)
        add_status = gr.Textbox(label="", lines=2, show_label=False)

        gr.HTML('<hr class="divider">')
        gr.Markdown("#### Reference Images of Matched Individual")
        reference_gallery = gr.Gallery(label="Reference images", columns=4, height="auto")

        identify_btn.click(
            fn=identify_leopard,
            inputs=input_image,
            outputs=[output_text, output_image, reference_gallery]
        )

        add_btn.click(
            fn=add_new_tiger,
            inputs=[tiger_name_input, input_image],
            outputs=add_status
        )

    # ── Webcam ───────────────────────────────────────────────────────────────
    with gr.Tab("\U0001F4F7 Webcam"):
        gr.HTML(f"""
        <div class="info-note">
            <span class="status-dot"></span>
            Live webcam identification &nbsp;\u00b7&nbsp;
            Threshold: <strong>{WEBCAM_THRESHOLD:.0%}</strong> &nbsp;\u00b7&nbsp;
            Preprocessing + Multi-crop enabled for phone screen captures
        </div>
        """)

        gr.HTML("""
        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem;">
            <span class="pipeline-step"><span class="step-icon">\U0001F4F8</span> Webcam Capture</span>
            <span class="pipeline-step"><span class="step-icon">\u2728</span> Preprocessing</span>
            <span class="pipeline-step"><span class="step-icon">\U0001F42F</span> Tiger Detection</span>
            <span class="pipeline-step"><span class="step-icon">\U0001F3AF</span> ID Matching</span>
        </div>
        """)

        gr.Markdown("#### Capture & Identify")
        with gr.Row():
            with gr.Column(scale=1):
                webcam_input = gr.Image(
                    label="Webcam Feed",
                    sources=["webcam"],
                    type="numpy"
                )
                webcam_btn = gr.Button("\U0001F4F7 Identify from Webcam", variant="primary")
            with gr.Column(scale=1):
                webcam_result = gr.Textbox(label="Identification Result", lines=14)
                webcam_processed = gr.Image(label="Processed Frame", visible=True)

        gr.HTML('<hr class="divider">')
        gr.Markdown("#### Reference Images of Matched Individual")
        webcam_gallery = gr.Gallery(label="Reference images", columns=4, height="auto")

        webcam_btn.click(
            fn=identify_from_webcam,
            inputs=webcam_input,
            outputs=[webcam_result, webcam_processed, webcam_gallery]
        )

    # ── Batch ────────────────────────────────────────────────────────────────
    with gr.Tab("\U0001F4CA Batch"):
        gr.HTML("""
        <div class="info-note">
            <span class="status-dot"></span>
            Process all images in <code>data/test/</code> against the reference database in one pass.
        </div>
        """)
        gr.Markdown("#### Batch Identification")
        batch_btn    = gr.Button("\u25b6 Run Batch Analysis", variant="primary")
        batch_output = gr.Textbox(label="Batch Results", lines=22)
        batch_btn.click(fn=batch_identify, outputs=batch_output)

    # ── Info ──────────────────────────────────────────────────────────────────
    with gr.Tab("\U0001F4D6 Info"):
        gr.HTML("""
        <div style="text-align: center; padding: 1.5rem 0 0.5rem 0;">
            <span style="font-size: 2.2rem;">\U0001F42F</span>
            <h2 style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem;
                       font-weight: 600; color: #fafafa; margin: 0.4rem 0 0.2rem 0;
                       letter-spacing: 0.06em; text-transform: uppercase;">
                How WildID Works
            </h2>
            <p style="font-size: 0.8rem; color: #71717a;">
                Deep learning pipeline for individual tiger identification via stripe pattern matching
            </p>
        </div>
        """)

        gr.HTML("""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;
                    margin: 1.2rem 0;">
            <div class="stat-card">
                <span class="stat-value">ViT-B/16</span>
                <span class="stat-label">Backbone</span>
            </div>
            <div class="stat-card">
                <span class="stat-value">128-D</span>
                <span class="stat-label">Embedding</span>
            </div>
            <div class="stat-card">
                <span class="stat-value">96%</span>
                <span class="stat-label">Threshold</span>
            </div>
            <div class="stat-card">
                <span class="stat-value">33</span>
                <span class="stat-label">Ref. Tigers</span>
            </div>
        </div>
        """)

        gr.Markdown("""
#### Method

- **Backbone** \u2014 ViT-Base (`vit_base_patch16_224`), last 4 transformer blocks fine-tuned
- **Training** \u2014 Triplet loss on 74 tiger identities (70/30 split, seed 42)
- **Embedding** \u2014 128-dimensional, L2 normalized feature vector
- **Matching** \u2014 Cosine similarity against reference database
- **Species Filter** \u2014 MobileNetV3 pre-check rejects non-tiger inputs
- **Webcam Pipeline** \u2014 Contrast/sharpness enhancement + median denoise + multi-crop TTA

#### Dataset Structure

```
data/
\u251c\u2500\u2500 reference/             \u2190 33 holdout tigers (known identities)
\u251c\u2500\u2500 fine_tuned_tigers/     \u2190 74 training tigers (not used at inference)
\u2514\u2500\u2500 test/                  \u2190 query images
```
        """)

if __name__ == "__main__":
    initialize_system()
    demo.launch(share=False, server_name="127.0.0.1", server_port=7863, css=css)

