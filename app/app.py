"""
Stage 5 - Web Application (Streamlit)

Four-tab interface:
  Tab 1 — Solve Equation       (primary user-facing solver)
  Tab 2 — Processing Details   (CV demonstration — most important for assessment)
  Tab 3 — History & Statistics (solved equation history with interactive charts)
  Tab 4 — Model Information    (training artefacts and CNN performance metrics)
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title='AI Handwritten Math Equation Solver',
    page_icon='🧮',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Pipeline loaders (each cached independently) ─────────────────────────────
@st.cache_resource(show_spinner='Loading CNN pipeline …')
def _load_cnn(model_path: str, threshold: float):
    from src.pipeline import MathSolverPipeline
    return MathSolverPipeline(model_path=model_path, confidence_threshold=threshold)

@st.cache_resource(show_spinner='Loading MobileNetV2 pipeline …')
def _load_mn(model_path: str, threshold: float):
    from src.pipeline import MathSolverPipeline
    return MathSolverPipeline(model_path=model_path, confidence_threshold=threshold)

# ── Utility: load image bytes → numpy BGR ─────────────────────────────────────
def _bytes_to_bgr(file_bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# ── Utility: numpy BGR → displayable RGB ──────────────────────────────────────
def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('⚙️ Settings')
    threshold = st.slider('Confidence Threshold', 0.5, 1.0, float(config.CONFIDENCE_THRESHOLD), 0.05)
    show_steps    = st.checkbox('Show preprocessing steps',   value=True)
    show_gradcam  = st.checkbox('Show Grad-CAM heatmaps',    value=True)

    st.divider()
    st.markdown('### Model Selection')
    _mn_exists  = config.MOBILENET_MODEL_PATH.exists()
    _cnn_exists = (config.CNN_MODEL_PATH.exists() or config.MODEL_PATH.exists())
    _model_options = ['CNN v3']
    if _mn_exists:
        _model_options += ['MobileNetV2', 'Both (Compare)']
    else:
        st.caption('MobileNetV2 not trained yet — train in Colab to enable.')
    model_choice = st.radio('Active model', _model_options, index=0)

    st.divider()
    st.markdown('### About')
    st.markdown(
        'AI Handwritten Math Equation Solver  \n'
        'Computer Vision University Project  \n'
        '**Stage 5 — Web Interface**'
    )
    st.markdown('**Team Members**')
    st.markdown('- [Your Name]  \n- [Team Member 2]  \n- [Team Member 3]')

# ── Load pipeline(s) ──────────────────────────────────────────────────────────
_cnn_path = str(config.CNN_MODEL_PATH if config.CNN_MODEL_PATH.exists() else config.MODEL_PATH)
pipeline  = _load_cnn(_cnn_path, threshold)   # always loaded (primary / fallback)

mn_pipeline = None
if model_choice in ('MobileNetV2', 'Both (Compare)') and _mn_exists:
    mn_pipeline = _load_mn(str(config.MOBILENET_MODEL_PATH), threshold)

# For single-model modes keep a unified reference
active_pipeline = mn_pipeline if (model_choice == 'MobileNetV2' and mn_pipeline) else pipeline

# ── Application header ────────────────────────────────────────────────────────
st.title('🧮 AI Handwritten Math Equation Solver')
st.markdown('*Upload a photo of a handwritten equation — get a step-by-step solution.*')
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    '📷 Solve Equation',
    '🔬 Processing Details',
    '📊 History & Statistics',
    '🧠 Model Information',
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Solve Equation
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header('Solve a Handwritten Equation')

    input_method = st.radio(
        'Choose input method:',
        ['📁 Upload Image', '📸 Use Camera', '✏️ Draw Equation'],
        horizontal=True
    )

    image_bgr    = None
    image_source = None

    if input_method == '📁 Upload Image':
        uploaded = st.file_uploader('Upload equation photo',
                                    type=['jpg', 'jpeg', 'png'])
        if uploaded:
            file_bytes = uploaded.read()
            image_bgr  = _bytes_to_bgr(file_bytes)
            st.image(_bgr_to_rgb(image_bgr), caption='Uploaded image', width="stretch")
            image_source = 'upload'

    elif input_method == '📸 Use Camera':
        camera_image = st.camera_input('Take a photo of your equation')
        if camera_image:
            file_bytes = camera_image.read()
            image_bgr  = _bytes_to_bgr(file_bytes)
            image_source = 'camera'

    elif input_method == '✏️ Draw Equation':
        try:
            from streamlit_drawable_canvas import st_canvas
            canvas_result = st_canvas(
                fill_color='rgba(0,0,0,0)',
                stroke_width=6,
                stroke_color='#000000',
                background_color='#ffffff',
                width=600, height=150,
                drawing_mode='freedraw',
                key='canvas',
            )
            if canvas_result.image_data is not None:
                rgba = canvas_result.image_data.astype(np.uint8)
                # Composite RGBA onto white background before converting.
                # Transparent pixels (alpha=0) become white (paper), not black (ink).
                alpha_f = rgba[:, :, 3:4].astype(np.float32) / 255.0
                rgb_f   = rgba[:, :, :3].astype(np.float32)
                white   = np.full_like(rgb_f, 255.0)
                composited = (rgb_f * alpha_f + white * (1.0 - alpha_f)).astype(np.uint8)
                image_bgr  = cv2.cvtColor(composited, cv2.COLOR_RGB2BGR)
                image_source = 'canvas'
        except ImportError:
            st.warning('Install `streamlit-drawable-canvas` to enable drawing.  \n'
                       '`pip install streamlit-drawable-canvas`')

    st.divider()

    # ── Solve button ───────────────────────────────────────────────────────
    solve_clicked = st.button('🔍 Solve Equation', type='primary',
                               disabled=(image_bgr is None and input_method != '✏️ Draw Equation'))

    if solve_clicked and image_bgr is not None:
        progress = st.progress(0, text='Starting pipeline …')

        try:
            if model_choice == 'Both (Compare)' and mn_pipeline:
                # ── Dual-model comparison mode ─────────────────────────────
                progress.progress(20, text='Running CNN …')
                cnn_result = pipeline.solve_from_image(image_bgr)
                progress.progress(60, text='Running MobileNetV2 …')
                mn_result  = mn_pipeline.solve_from_image(image_bgr)
                progress.progress(100, text='Done!')

                # Pick the higher-confidence result as primary (stored for Tab 2)
                cnn_conf = cnn_result.get('recognition', {}).get('confidence', 0.0)
                mn_conf  = mn_result.get('recognition',  {}).get('confidence', 0.0)
                primary_result = cnn_result if cnn_conf >= mn_conf else mn_result
                st.session_state['last_result'] = primary_result
                st.session_state['last_image']  = image_bgr

                st.subheader('Model Comparison')
                col_cnn, col_mn = st.columns(2)

                for col, res, name, conf in [
                    (col_cnn, cnn_result, 'CNN v3',      cnn_conf),
                    (col_mn,  mn_result,  'MobileNetV2', mn_conf),
                ]:
                    with col:
                        parsing  = res.get('parsing',  {})
                        solution = res.get('solution', {})
                        raw_sol  = solution.get('raw', {})

                        winner_badge = ' 🏆' if (
                            (name == 'CNN v3' and cnn_conf >= mn_conf) or
                            (name == 'MobileNetV2' and mn_conf > cnn_conf)
                        ) else ''
                        st.markdown(f"### {name}{winner_badge}")
                        st.metric('Confidence', f'{conf*100:.1f}%')

                        eq_str = parsing.get('equation_str', '')
                        if eq_str:
                            st.info(f'`{eq_str}`')

                        if solution.get('success'):
                            st.success(raw_sol.get('answer_str', '—'))
                        else:
                            st.error(solution.get('error', 'Failed'))

                        sym_results = res.get('recognition', {}).get('symbol_results', [])
                        if sym_results:
                            syms = ' '.join(sr.get('symbol', '?') for sr in sym_results)
                            st.caption(f'Recognised: `{syms}`')

                        with st.expander('Steps'):
                            for step in raw_sol.get('steps', []):
                                st.markdown(
                                    f"**{step['step_number']}. {step['description']}**  \n"
                                    f"`{step['expression']}`  \n"
                                    f"*{step.get('explanation', '')}*"
                                )

            else:
                # ── Single-model mode ──────────────────────────────────────
                progress.progress(10, text='Preprocessing image …')
                progress.progress(30, text='Segmenting characters …')
                progress.progress(55, text='Recognising symbols …')
                progress.progress(75, text='Parsing equation …')
                progress.progress(90, text='Solving equation …')

                result = active_pipeline.solve_from_image(image_bgr)
                progress.progress(100, text='Done!')

                st.session_state['last_result'] = result
                st.session_state['last_image']  = image_bgr

                st.subheader('Results')
                parsing  = result.get('parsing',  {})
                solution = result.get('solution', {})
                recog    = result.get('recognition', {})

                if not parsing.get('success'):
                    st.error(f"Parsing failed: {parsing.get('error', 'Unknown error')}")
                elif not solution.get('success'):
                    st.error(f"Solving failed: {solution.get('error', 'Unknown error')}")
                else:
                    eq_str = parsing.get('equation_str', 'N/A')
                    st.info(f'**Recognised equation:** `{eq_str}`')

                    raw_sol = solution.get('raw', {})
                    st.success(f"### {raw_sol.get('answer_str', 'No answer')}")

                    conf = recog.get('confidence', 0.0)
                    st.metric('Recognition Confidence', f"{conf*100:.1f}%")
                    if conf < 0.60:
                        st.warning(
                            '⚠️ Confidence is below 60%.  \n'
                            'Retake the photo with better lighting and clearer writing.'
                        )

                    with st.expander('📐 Step-by-step solution', expanded=True):
                        for step in raw_sol.get('steps', []):
                            st.markdown(
                                f"**Step {step['step_number']}**: {step['description']}  \n"
                                f"`{step['expression']}`  \n"
                                f"*{step.get('explanation', '')}*"
                            )
                            st.divider()

                    try:
                        report_path = active_pipeline.generate_report(result)
                        with open(report_path, 'r') as f:
                            st.download_button(
                                '⬇️ Download Full Report (HTML)',
                                data=f.read(),
                                file_name=report_path.name,
                                mime='text/html'
                            )
                    except Exception:
                        pass

        except Exception as e:
            progress.empty()
            st.error(f'Pipeline error: {e}')

    elif not active_pipeline.model_ready:
        st.warning(
            '⚠️ CNN model has not been trained yet.  \n'
            'Run:  `python main.py --train`  then restart the app.'
        )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Processing Details (primary CV demonstration tab)
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header('Processing Details')
    st.markdown('*This tab demonstrates every classical CV and deep learning step.*')

    result     = st.session_state.get('last_result', None)
    last_image = st.session_state.get('last_image', None)

    if result is None:
        st.info('Solve an equation in the **Solve Equation** tab first.')
    else:
        # ── Section 1: Classical CV Preprocessing ─────────────────────────
        st.subheader('1️⃣ Classical CV Preprocessing')
        st.markdown(
            'Each operation cleans the raw photo one step at a time. '
            'Together they ensure the CNN receives a consistent, noise-free input.'
        )

        pipeline_steps = result.get('preprocessing', {}).get('pipeline_steps', {})
        step_captions = {
            '1_original':    ('Original', 'Raw camera photo — colour, noise, tilt'),
            '2_grayscale':   ('Grayscale', 'Colour removed — only ink intensity needed'),
            '3_blurred':     ('Gaussian Blur', 'Camera noise & paper texture suppressed'),
            '4_binary_otsu': ('Otsu Threshold', 'Adaptive binarisation — ink=white, paper=black'),
            '5_deskewed':    ('Deskewed', 'Text angle detected & corrected to horizontal'),
            '6_opened':      ('Morph. Opening', 'Noise dots removed via erosion+dilation'),
            '7_closed':      ('Morph. Closing', 'Character gaps filled via dilation+erosion'),
            '8_normalized':  ('Normalised', 'Values scaled to [0,1] for CNN input'),
        }

        if pipeline_steps and show_steps:
            cols = st.columns(len(pipeline_steps))
            for col, (step_key, step_img) in zip(cols, pipeline_steps.items()):
                with col:
                    title, caption = step_captions.get(step_key, (step_key, ''))
                    if isinstance(step_img, np.ndarray):
                        display = step_img
                        if step_img.dtype != np.uint8:
                            display = (step_img * 255).astype(np.uint8)
                        st.image(_bgr_to_rgb(display), caption=f'**{title}**', width="stretch")
                    st.caption(caption)
        else:
            st.info('Enable "Show preprocessing steps" in the sidebar.')

        # ── Section 2: Character Segmentation ────────────────────────────
        st.subheader('2️⃣ Character Segmentation')
        st.markdown(
            'Contour detection (cv2.findContours) isolates each ink region. '
            'Green boxes = normal, Red = superscript, Blue = subscript.'
        )

        seg = result.get('segmentation', {})
        annotated = seg.get('annotated_image')
        characters = seg.get('characters', [])

        if annotated is not None:
            st.image(_bgr_to_rgb(annotated), caption='Detected characters with bounding boxes',
                     width="stretch")
            st.metric('Characters detected', seg.get('character_count', 0))

        if characters:
            st.markdown('**Individual character crops (32×32):**')
            crop_cols = st.columns(min(len(characters), 10))
            for i, (col, ch) in enumerate(zip(crop_cols, characters[:10])):
                with col:
                    st.image(ch['image'], caption=f"#{i}\n{ch['position_type'][:3].upper()}",
                             width="stretch", clamp=True)

        # ── Section 3: Deep Learning Recognition ──────────────────────────
        st.subheader('3️⃣ Deep Learning Recognition')
        rec = result.get('recognition', {})
        sym_results = rec.get('symbol_results', [])

        if not rec.get('success'):
            st.warning(f"Recognition: {rec.get('error', 'CNN model not available.')}")
        elif sym_results:
            cam_overlays = rec.get('gradcam_overlays', [])

            for i, (ch, sr) in enumerate(zip(characters, sym_results)):
                with st.expander(
                    f"Character #{i}  →  '{sr.get('symbol','?')}'  "
                    f"({sr.get('confidence',0)*100:.0f}%)"
                ):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(ch['image'], caption='Crop', width="stretch", clamp=True)
                        st.markdown(f"**Predicted: `{sr.get('symbol','?')}`**")
                    with c2:
                        # Top-3 bar chart
                        top3 = sr.get('top3', [])
                        fig, ax = plt.subplots(figsize=(3, 1.5))
                        syms  = [t['symbol']     for t in top3]
                        confs = [t['confidence'] for t in top3]
                        colors = ['green' if c >= 0.9 else 'orange' if c >= 0.7 else 'red'
                                  for c in confs]
                        ax.barh(syms, confs, color=colors)
                        ax.set_xlim(0, 1)
                        ax.set_xlabel('Probability')
                        ax.set_title('Top-3 predictions')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    with c3:
                        if show_gradcam and i < len(cam_overlays):
                            overlay = cam_overlays[i]
                            st.image(_bgr_to_rgb(overlay), caption='Grad-CAM',
                                     width="stretch")
                        else:
                            st.caption('Enable Grad-CAM in sidebar')

        # ── Section 4: Equation Parsing ───────────────────────────────────
        st.subheader('4️⃣ Equation Parsing')
        st.markdown('Each transformation rule converts raw symbol tokens to a valid equation.')

        parsing = result.get('parsing', {})
        tx_steps = parsing.get('transformation_steps', {})

        step_labels = {
            'raw':               '① Raw symbols from CNN',
            'after_exponents':   '② After superscript → ** (exponents)',
            'after_digit_merge': '③ After consecutive digit merging',
            'after_implicit_mult': '④ After implicit multiply insertion (2x → 2*x)',
            'after_negatives':   '⑤ After negative number handling',
        }
        for key, label in step_labels.items():
            if key in tx_steps:
                tokens = tx_steps[key]
                st.markdown(f"**{label}**")
                st.code(str(tokens), language='python')

        if parsing.get('equation_str'):
            st.success(f"**Final equation:**  `{parsing['equation_str']}`")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — History & Statistics
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header('History & Statistics')

    history = active_pipeline.history
    records = history.records

    if not records:
        st.info('No equations solved yet. Solve some equations in the Solve tab!')
    else:
        try:
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go

            df = pd.DataFrame([{
                'timestamp':     r.get('timestamp', '')[:19],
                'equation':      r.get('equation', ''),
                'type':          r.get('equation_type', ''),
                'confidence':    round(float(r.get('confidence', 0)) * 100, 1),
                'answer':        r.get('solution', {}).get('answer_str', '')
                                  if isinstance(r.get('solution'), dict) else '',
            } for r in records])

            st.dataframe(df, width="stretch")

            col1, col2, col3 = st.columns(3)
            col1.metric('Total Solved', len(records))
            col2.metric('Avg Confidence', f"{df['confidence'].mean():.1f}%")
            col3.metric('Unique Types', df['type'].nunique())

            st.divider()

            # Pie chart — equation type distribution
            type_counts = df['type'].value_counts().reset_index()
            type_counts.columns = ['type', 'count']
            fig_pie = px.pie(type_counts, names='type', values='count',
                             title='Equation Type Distribution',
                             color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_pie, width="stretch")

            # Bar chart — confidence distribution
            fig_conf = px.histogram(df, x='confidence', nbins=10,
                                    title='Confidence Score Distribution',
                                    labels={'confidence': 'Confidence (%)'},
                                    color_discrete_sequence=['steelblue'])
            st.plotly_chart(fig_conf, width="stretch")

            # Line chart — equations over time
            if len(df) > 1:
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
                daily = df.groupby('date').size().reset_index(name='count')
                fig_line = px.line(daily, x='date', y='count',
                                   title='Equations Solved Over Time',
                                   markers=True)
                st.plotly_chart(fig_line, width="stretch")

            # CSV export
            csv_path = history.export_csv()
            with open(csv_path, 'r') as f:
                st.download_button('⬇️ Download History (CSV)', f.read(),
                                   file_name='equation_history.csv', mime='text/csv')

        except ImportError as e:
            st.warning(f"Install pandas and plotly for charts: {e}")
            for r in records:
                st.write(f"• `{r.get('equation','')}` → {r.get('solution',{}).get('answer_str','')}")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Information
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header('Model Information')
    import io as _io

    # ── Helper: show image or placeholder ─────────────────────────────────────
    def _show_img(path, caption=''):
        if path.exists():
            st.image(str(path), caption=caption, width="stretch")
        else:
            st.info(f'Not found: `{path.name}`  \nRun Colab training then place the file in `models/`.')

    # ── 1. Model Comparison (dual-train artefacts) ────────────────────────────
    st.subheader('1️⃣  CNN  vs  MobileNetV2 — Comparison')
    if config.MODEL_COMPARISON_IMG.exists() or config.CONFUSION_MATRICES_IMG.exists():
        cmp_c1, cmp_c2 = st.columns(2)
        with cmp_c1:
            _show_img(config.MODEL_COMPARISON_IMG, 'Training curves — accuracy & loss')
        with cmp_c2:
            _show_img(config.CONFUSION_MATRICES_IMG, 'Confusion matrices')
    else:
        st.info(
            'Comparison charts not generated yet.  \n'
            'Run the Colab notebook (dual-train) and copy the downloaded PNGs to `models/`:\n'
            '- `models/model_comparison.png`\n'
            '- `models/confusion_matrices.png`'
        )

    st.divider()

    # ── 2. Per-class Accuracy ─────────────────────────────────────────────────
    st.subheader('2️⃣  Per-class Accuracy')
    _show_img(config.PER_CLASS_ACC_IMG, 'Per-class accuracy — CNN vs MobileNetV2')

    st.divider()

    # ── 3. Sample Predictions ─────────────────────────────────────────────────
    st.subheader('3️⃣  Sample Predictions')
    _show_img(config.PREDICTION_SAMPLES_IMG, 'Test-set sample predictions')

    st.divider()

    # ── 4. Individual training histories ─────────────────────────────────────
    st.subheader('4️⃣  Training Histories')
    hist_c1, hist_c2 = st.columns(2)
    with hist_c1:
        # Prefer dual-train artefact, fall back to legacy
        p = config.CNN_TRAINING_HIST_IMG if config.CNN_TRAINING_HIST_IMG.exists() else config.TRAINING_HIST_IMG
        _show_img(p, 'CNN v3 — accuracy & loss curves')
    with hist_c2:
        _show_img(config.MN_TRAINING_HIST_IMG, 'MobileNetV2 — accuracy & loss curves')

    st.divider()

    # ── 5. Grad-CAM ───────────────────────────────────────────────────────────
    st.subheader('5️⃣  Grad-CAM — Where the Model Looks')
    st.markdown(
        'Red = high attention (decisive pixels). Blue = ignored.  \n'
        'A well-trained model should highlight the characteristic strokes of each symbol.'
    )

    _show_img(config.GRADCAM_CMP_IMG, 'Side-by-side: CNN vs MobileNetV2 attention (first 8 classes)')

    gc_c1, gc_c2 = st.columns(2)
    with gc_c1:
        _show_img(config.GRADCAM_CNN_IMG, 'CNN v3 — all 16 classes')
    with gc_c2:
        _show_img(config.GRADCAM_MN_IMG, 'MobileNetV2 — all 16 classes')

    st.divider()

    # ── 5b. Legacy single-run artefacts ──────────────────────────────────────
    if config.CONFUSION_MAT_IMG.exists() or config.GRADCAM_IMG.exists():
        with st.expander('Legacy single-model artefacts'):
            lc1, lc2 = st.columns(2)
            with lc1:
                _show_img(config.CONFUSION_MAT_IMG, 'Confusion matrix (single run)')
            with lc2:
                _show_img(config.GRADCAM_IMG, 'Grad-CAM (single run)')

    st.divider()

    # ── 6. Architecture summaries ─────────────────────────────────────────────
    st.subheader('6️⃣  Architecture Summaries')
    _models_to_show = [('CNN v3', pipeline)]
    if mn_pipeline and mn_pipeline.model_ready:
        _models_to_show.append(('MobileNetV2', mn_pipeline))

    _sum_cols = st.columns(len(_models_to_show))
    for col, (mname, mpipe) in zip(_sum_cols, _models_to_show):
        with col:
            st.markdown(f'**{mname}**')
            if mpipe.model_ready and mpipe.recognizer:
                buf = _io.StringIO()
                mpipe.recognizer.model.summary(print_fn=lambda x: buf.write(x + '\n'))
                st.code(buf.getvalue(), language='')
            else:
                st.info('Model not loaded.')

    st.divider()

    # ── 7. Dataset statistics ─────────────────────────────────────────────────
    st.subheader('7️⃣  Dataset Statistics')
    processed_dir = config.PROCESSED_DIR
    _any_split = False
    for split_name in ('train', 'val', 'test'):
        labels_path = processed_dir / f'{split_name}_labels.npy'
        if labels_path.exists():
            _any_split = True
            lbls = np.load(str(labels_path))
            unique, counts = np.unique(lbls, return_counts=True)
            rows = [{'Class': config.CLASS_MAP.get(int(c), str(c)),
                     'Samples': int(cnt)} for c, cnt in zip(unique, counts)]
            try:
                import pandas as pd
                st.markdown(f'**{split_name.capitalize()} set** ({len(lbls):,} samples)')
                st.dataframe(pd.DataFrame(rows), width="stretch")
            except ImportError:
                st.write({r['Class']: r['Samples'] for r in rows})
    if not _any_split:
        st.info('No processed dataset found. Run training to generate split statistics.')
