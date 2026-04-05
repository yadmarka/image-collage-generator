"""
Streamlit web app for the Image Collage Generator.
Allows users to upload a target image and source images, tune parameters,
and generate a photomosaic collage in the browser.
"""

import io
import hashlib
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.categorize_images import SourceImagePalette, categorize_single_image

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Image Collage Generator",
    page_icon="🎨",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
for key, default in [
    ("palette", None),
    ("palette_key", None),      # hash of source files to detect changes
    ("palette_tmpdir", None),   # temp dir holding source files for rendering
    ("collage", None),
    ("target_image", None),
    ("processing_time", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _files_hash(uploaded_files) -> str:
    """Compute a deterministic hash over a list of UploadedFile objects."""
    h = hashlib.md5()
    for f in sorted(uploaded_files, key=lambda x: x.name):
        h.update(f.name.encode())
        h.update(str(f.size).encode())
    return h.hexdigest()


def build_palette(uploaded_files, status_placeholder):
    """
    Save uploaded files to a temp directory and build a SourceImagePalette.

    The temp directory is kept alive so render_collage can open the files
    later via SourceImage.filepath.  The caller is responsible for cleaning
    up the previous tmpdir before calling this function.

    Returns (SourceImagePalette, tmpdir_path).
    """
    import shutil

    palette = SourceImagePalette()
    tmpdir = tempfile.mkdtemp(prefix="collage_sources_")
    failed = []

    for i, f in enumerate(uploaded_files):
        status_placeholder.info(
            f"⚙️ Processing source image {i + 1}/{len(uploaded_files)}: **{f.name}**"
        )
        tmp_path = os.path.join(tmpdir, f.name)
        with open(tmp_path, "wb") as fh:
            fh.write(f.getbuffer())
        try:
            src_img = categorize_single_image(tmp_path)
            palette.add_image(src_img)
        except Exception as e:
            failed.append((f.name, str(e)))
            # Remove the file we just wrote if it couldn't be processed
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if failed:
        st.warning(
            f"⚠️ {len(failed)} source image(s) could not be processed: "
            + ", ".join(name for name, _ in failed)
        )

    return palette, tmpdir


def render_collage(
    target: Image.Image,
    palette: SourceImagePalette,
    num_cols: int,
    num_rows: int,
    progress_bar,
) -> Image.Image:
    """
    Render the photomosaic.

    Divides *target* into a num_cols × num_rows grid, finds the closest
    source image for each tile by average RGB colour, and assembles the
    final mosaic.  Tile dimensions are computed from the image size so
    the output is the same resolution as the (cropped) target.
    """
    width, height = target.size

    tile_w = width // num_cols
    tile_h = height // num_rows

    if tile_w == 0 or tile_h == 0:
        raise ValueError(
            f"Grid {num_cols}×{num_rows} is too fine for a "
            f"{width}×{height} image. Reduce grid size."
        )

    # Crop to a clean multiple of tile dimensions
    cropped_w = tile_w * num_cols
    cropped_h = tile_h * num_rows
    target = target.crop((0, 0, cropped_w, cropped_h))

    mosaic = Image.new("RGB", (cropped_w, cropped_h))
    total_tiles = num_cols * num_rows
    done = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x = col * tile_w
            y = row * tile_h

            tile = target.crop((x, y, x + tile_w, y + tile_h))
            avg_color = tuple(np.array(tile).mean(axis=(0, 1)).astype(int))

            best = palette.find_closest_match(avg_color)

            # Load source image from its saved filepath
            src = Image.open(best.filepath).convert("RGB")
            src = src.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            mosaic.paste(src, (x, y))

            done += 1
            progress_bar.progress(done / total_tiles)

    return mosaic


def pil_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Encode a PIL Image to bytes for download."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def show_image_previews(uploaded_files, max_cols: int = 8):
    """Display thumbnail previews for a list of UploadedFile objects."""
    cols = st.columns(min(len(uploaded_files), max_cols))
    for i, f in enumerate(uploaded_files):
        with cols[i % max_cols]:
            img = Image.open(f)
            img.thumbnail((120, 120))
            st.image(img, caption=f.name, use_container_width=False)


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.title("🎨 Image Collage Generator")
st.markdown(
    """
    Turn any photo into a **photomosaic** — a collage built entirely from your
    own source images, each chosen by colour similarity.

    **How to use:**
    1. Upload a **target image** (the photo you want to recreate).
    2. Upload **source images** (the "tiles" — more variety = better results).
    3. Adjust the grid and matching settings in the sidebar.
    4. Click **Generate Collage** and download your result.
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Sidebar — parameters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    grid_cols = st.slider("Grid columns", min_value=10, max_value=100, value=40, step=1,
                          help="Number of tile columns across the output image.")
    grid_rows = st.slider("Grid rows",    min_value=10, max_value=100, value=30, step=1,
                          help="Number of tile rows down the output image.")

    with st.expander("Advanced settings"):
        match_method = st.selectbox(
            "Colour matching method",
            options=["Euclidean (faster)", "Delta E (perceptual)"],
            index=0,
            help=(
                "**Euclidean** computes distance in RGB space — fast and usually good.\n\n"
                "**Delta E** uses perceptual CIE LAB space — slower but more accurate."
            ),
        )

    st.divider()
    st.markdown("**Minimum recommended sources:** 20+  \nMore unique colours → better collage.")

# ---------------------------------------------------------------------------
# Main columns — uploaders
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🖼️ Target image")
    target_file = st.file_uploader(
        "Upload the photo to recreate",
        type=["jpg", "jpeg", "png", "webp"],
        key="target_uploader",
    )

    if target_file:
        target_pil = Image.open(target_file).convert("RGB")
        st.session_state.target_image = target_pil
        st.image(target_pil, caption=f"{target_file.name}  ({target_pil.size[0]}×{target_pil.size[1]}px)", use_container_width=True)

with col_right:
    st.subheader("🗂️ Source images (tiles)")
    source_files = st.file_uploader(
        "Upload the images that will become tiles",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="source_uploader",
    )

    if source_files:
        st.caption(f"{len(source_files)} image(s) uploaded")
        show_image_previews(source_files, max_cols=8)

st.divider()

# ---------------------------------------------------------------------------
# Validation & Generate button
# ---------------------------------------------------------------------------
ready = bool(target_file and source_files)

if not target_file:
    st.info("ℹ️ Upload a target image to get started.")
elif not source_files:
    st.info("ℹ️ Upload at least one source image.")
elif len(source_files) < 5:
    st.warning("⚠️ Fewer than 5 source images will produce a low-quality collage. Upload more for better results.")

generate_clicked = st.button(
    "✨ Generate Collage",
    type="primary",
    disabled=not ready,
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Generation pipeline
# ---------------------------------------------------------------------------
if generate_clicked and ready:
    status = st.empty()
    start_time = time.time()

    try:
        # --- Step 1: Build / reuse palette ---
        current_key = _files_hash(source_files)
        if st.session_state.palette_key != current_key:
            # Clean up the previous temp directory before creating a new one
            if st.session_state.palette_tmpdir:
                import shutil
                shutil.rmtree(st.session_state.palette_tmpdir, ignore_errors=True)
                st.session_state.palette_tmpdir = None

            status.info("🔍 Processing source images…")
            with st.spinner("Building colour palette from source images…"):
                palette, tmpdir = build_palette(source_files, status)

            if len(palette) == 0:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
                st.error("❌ No source images could be processed. Check that your files are valid images.")
                st.stop()

            st.session_state.palette = palette
            st.session_state.palette_key = current_key
            st.session_state.palette_tmpdir = tmpdir  # keep files alive for rendering
        else:
            palette = st.session_state.palette
            st.info(f"♻️ Reusing palette from previous run ({len(palette)} images).")

        # --- Step 2: Render ---
        status.info("🎨 Matching colours and rendering mosaic…")
        progress_bar = st.progress(0)

        target_pil = st.session_state.target_image
        collage = render_collage(target_pil, palette, grid_cols, grid_rows, progress_bar)

        elapsed = time.time() - start_time
        st.session_state.collage = collage
        st.session_state.processing_time = elapsed

        progress_bar.empty()
        status.success(f"✅ Collage generated in {elapsed:.1f}s!")

    except ValueError as e:
        st.error(f"❌ Configuration error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error during generation: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
if st.session_state.collage is not None:
    st.divider()
    st.subheader("🖼️ Result")

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    palette = st.session_state.palette
    collage = st.session_state.collage
    target_pil = st.session_state.target_image

    c1.metric("Source images", len(palette) if palette else "—")
    c2.metric("Grid size", f"{grid_cols} × {grid_rows}")
    c3.metric("Output size", f"{collage.size[0]}×{collage.size[1]}px")
    c4.metric("Processing time", f"{st.session_state.processing_time:.1f}s" if st.session_state.processing_time else "—")

    # Side-by-side comparison
    tab_collage, tab_compare, tab_original = st.tabs(["Collage", "Side-by-side comparison", "Original"])

    with tab_collage:
        st.image(collage, caption="Generated photomosaic", use_container_width=True)

    with tab_compare:
        l, r = st.columns(2)
        with l:
            st.image(target_pil, caption="Original target", use_container_width=True)
        with r:
            st.image(collage, caption="Photomosaic collage", use_container_width=True)

    with tab_original:
        st.image(target_pil, caption="Original target image", use_container_width=True)

    # Download button
    st.download_button(
        label="⬇️ Download collage (PNG)",
        data=pil_to_bytes(collage, "PNG"),
        file_name="photomosaic_collage.png",
        mime="image/png",
        use_container_width=True,
        type="primary",
    )

    #streamlit run app.py