import gradio as gr
import os
import sys
import shutil
import time
from pathlib import Path

sys.path.append('src')

from feature_extractor import SnowLeopardFeatureExtractor
from matcher import SnowLeopardMatcher
from utils import (
    parse_reference_structure,
    save_features,
    load_features,
    validate_dataset
)

# Global variables
extractor = None
matcher = None
reference_features = {}

THRESHOLD = 0.96
PKL_PATH = 'models/reference_features.pkl'
REFERENCE_DIR = 'data/reference'

def initialize_system():
    global extractor, matcher
    extractor = SnowLeopardFeatureExtractor(device='cpu')
    matcher = SnowLeopardMatcher()

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
    global extractor, matcher, last_query_path

    if matcher is None or not matcher.reference_features:
        return "Load or build the reference database first.", None, []

    try:
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
                f"RESULT — NEW TIGER DETECTED\n"
                f"{'─'*40}\n\n"
                f"No match found in reference database.\n"
                f"Confidence threshold : {THRESHOLD:.0%}\n"
                f"Best score obtained  : {confidence:.2%}\n\n"
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


def add_new_tiger(tiger_name, image):
    """Add the query image as a new tiger into the reference database."""
    global extractor, matcher, reference_features, last_query_path

    tiger_name = tiger_name.strip()
    if not tiger_name:
        return "Enter a tiger name first."

    if image is None and last_query_path is None:
        return "No query image available to add."

    if matcher is None:
        return "Database not loaded. Load or build the database first."

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
    global extractor, matcher

    if matcher is None or not matcher.reference_features:
        return "Load or build the reference database first."

    from utils import load_images_from_directory
    test_images = load_images_from_directory(test_dir)

    if not test_images:
        return f"No test images found in {test_dir}"

    results = f"BATCH RESULTS — {len(test_images)} images\n{'─'*50}\n\n"

    for img_path in test_images:
        features = extractor.extract_features(img_path)
        if features is not None:
            best_match, confidence, _ = matcher.find_match(features, threshold=THRESHOLD)
            status = "NEW TIGER" if best_match == "Unknown" else best_match
            results += f"{os.path.basename(img_path)}\n"
            results += f"  → {status}  ({confidence:.2%})\n\n"

    return results


# ── CSS ───────────────────────────────────────────────────────────────────────
css = """
footer { display: none !important; }
.built-with { display: none !important; }
#svelte-announcer { display: none !important; }
.gradio-container > .footer { display: none !important; }

@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

body, .gradio-container {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background: #0e0e0e !important;
    color: #d4d4d4 !important;
}

.header-block {
    border-bottom: 1px solid #2a2a2a;
    padding: 2rem 0 1.5rem 0;
    margin-bottom: 1rem;
}

.header-block h1 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.3rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    color: #e8e8e8 !important;
    text-transform: uppercase !important;
    margin: 0 !important;
}

.tab-nav button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #555 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.6rem 1.2rem !important;
    transition: color 0.2s, border-color 0.2s !important;
}

.tab-nav button.selected {
    color: #d4d4d4 !important;
    border-bottom: 2px solid #d4d4d4 !important;
}

.tab-nav button:hover { color: #999 !important; }

button.primary {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    background: #1e1e1e !important;
    color: #d4d4d4 !important;
    border: 1px solid #333 !important;
    border-radius: 2px !important;
    padding: 0.6rem 1.4rem !important;
    transition: background 0.2s, border-color 0.2s !important;
    cursor: pointer !important;
}

button.primary:hover {
    background: #2a2a2a !important;
    border-color: #555 !important;
}

button.secondary {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    color: #666 !important;
    border: 1px solid #252525 !important;
    border-radius: 2px !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
}

button.secondary:hover {
    color: #aaa !important;
    border-color: #3a3a3a !important;
}

textarea, .gr-textbox textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    background: #141414 !important;
    border: 1px solid #222 !important;
    border-radius: 2px !important;
    color: #b0b0b0 !important;
    line-height: 1.7 !important;
}

input[type="text"], .gr-textbox input {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    background: #141414 !important;
    border: 1px solid #222 !important;
    border-radius: 2px !important;
    color: #b0b0b0 !important;
}

label span, .gr-form label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #555 !important;
}

.gr-image {
    border: 1px solid #222 !important;
    border-radius: 2px !important;
    background: #141414 !important;
}

.gr-gallery {
    background: #141414 !important;
    border: 1px solid #222 !important;
    border-radius: 2px !important;
}

.new-tiger-panel {
    border: 1px solid #2a2a2a;
    border-left: 3px solid #444;
    background: #111;
    padding: 1rem 1.2rem 0.5rem 1.2rem;
    margin-top: 1rem;
    border-radius: 2px;
}

.new-tiger-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #555;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0 0 0.8rem 0;
}

.gr-markdown h2, .gr-markdown h3, .gr-markdown h4 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #888 !important;
    font-weight: 500 !important;
    border-bottom: 1px solid #1e1e1e !important;
    padding-bottom: 0.4rem !important;
    margin-top: 1.4rem !important;
}

.gr-markdown p, .gr-markdown li {
    font-size: 0.82rem !important;
    color: #666 !important;
    line-height: 1.8 !important;
}

.gr-markdown code, .gr-markdown pre {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.74rem !important;
    background: #141414 !important;
    border: 1px solid #222 !important;
    color: #888 !important;
    border-radius: 2px !important;
}

.threshold-note {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #444;
    letter-spacing: 0.08em;
    margin: 0.3rem 0 1rem 0;
}

.divider {
    border: none;
    border-top: 1px solid #1e1e1e;
    margin: 1rem 0;
}
"""

# ── Interface ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Tiger Identification", css=css) as demo:

    gr.HTML("""
    <div class="header-block">
        <h1>Tiger Individual Identification System</h1>
    </div>
    """)

    # ── Setup ─────────────────────────────────────────────────────────────────
    with gr.Tab("Setup"):
        gr.Markdown("#### Database")

        with gr.Row():
            load_btn  = gr.Button("Load Existing Database",      variant="primary")
            build_btn = gr.Button("Build Database from Images",  variant="secondary")

        gr.HTML('<hr class="divider">')
        setup_output = gr.Textbox(label="Status", lines=14)

        load_btn.click(fn=load_existing_database, outputs=setup_output)

        # build_reference_database is a generator — Gradio streams each yield
        # into setup_output automatically
        build_btn.click(fn=build_reference_database, outputs=setup_output)

        gr.HTML('<hr class="divider">')
        gr.HTML('<p class="threshold-note">Delete a tiger — removes from live database, pkl, and reference folder</p>')
        with gr.Row():
            delete_name_input = gr.Textbox(
                label="Tiger Name / ID to Delete",
                placeholder="e.g.  tiger_030",
                scale=3
            )
            delete_btn = gr.Button("Delete Tiger", variant="secondary", scale=1)
        delete_output = gr.Textbox(label="", lines=6, show_label=False)

        delete_btn.click(
            fn=delete_tiger,
            inputs=delete_name_input,
            outputs=delete_output
        )

    # ── Identify ──────────────────────────────────────────────────────────────
    with gr.Tab("🔍 Identify"):
        gr.Markdown("#### Single Image Identification")
        gr.HTML(
            f'<p class="threshold-note">Similarity threshold: {THRESHOLD:.0%}'
            f' &nbsp;·&nbsp; Scores below this are flagged as new tiger</p>'
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image  = gr.Image(label="Query Image", type="pil")
                identify_btn = gr.Button("Identify", variant="primary")
            with gr.Column(scale=1):
                output_text  = gr.Textbox(label="Result", lines=14)
                output_image = gr.Image(visible=False)

        gr.HTML('<hr class="divider">')
        gr.HTML('<p class="threshold-note">New tiger detected? Name it and add to database</p>')
        with gr.Row():
            tiger_name_input = gr.Textbox(
                label="Tiger Name / ID",
                placeholder="e.g.  tiger_275",
                scale=3
            )
            add_btn = gr.Button("Add New Tiger to Database", variant="secondary", scale=1)
        add_status = gr.Textbox(label="", lines=2, show_label=False)

        gr.HTML('<hr class="divider">')
        gr.Markdown("#### Reference images of matched individual")
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

    # ── Batch ─────────────────────────────────────────────────────────────────
    with gr.Tab("Batch"):
        gr.Markdown("#### Batch Identification")
        gr.Markdown("Processes all images in `data/test/` against the reference database.")
        batch_btn    = gr.Button("Run Batch", variant="primary")
        batch_output = gr.Textbox(label="Batch Results", lines=22)
        batch_btn.click(fn=batch_identify, outputs=batch_output)

    # ── Info ──────────────────────────────────────────────────────────────────
    with gr.Tab("ℹ️ Info"):
        gr.Markdown("""
#### About

This system identifies individual tigers by matching stripe patterns using deep learning embeddings.

#### Method

- **Backbone** — ViT-Base (vit_base_patch16_224), last 4 blocks fine-tuned
- **Training** — Triplet loss on 74 tiger identities (70% split, seed 42)
- **Embedding** — 128-dimensional, L2 normalized
- **Matching** — Cosine similarity against reference database
- **Reference DB** — 33 holdout tigers (never seen during training)
- **Threshold** — 0.96 (scores below this → New Tiger Detected)

#### Dataset Structure

```
data/
├── reference/             ← 33 holdout tigers (known identities)
├── fine_tuned_tigers/     ← 74 training tigers (not used at inference)
└── test/                  ← query images
```
        """)

if __name__ == "__main__":
    initialize_system()
    demo.launch(share=False, server_name="127.0.0.1", server_port=7861)
