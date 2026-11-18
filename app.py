
import os
import html
import gradio as gr

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.evaluation import evaluate_answer


# =====================================================
# PREMIUM DARK-GLASS THEME CSS
# =====================================================

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

/* --- Global Reset & Background --- */
html, body, .gradio-container {
    background: #090b14 !important;
    background-color: #090b14 !important;
    color: #e2e8f0 !important;
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    margin: 0 auto !important;
    max-width: 1560px !important;
    width: 96% !important;
}

/* Force dark on all Gradio wrapper blocks */
.gradio-container * {
    box-sizing: border-box;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container fieldset,
.gradio-container .wrap,
.gradio-container .input-container,
.gradio-container div[data-testid="textbox"] {
    background: #111528 !important;
    background-color: #111528 !important;
    border-color: rgba(139, 92, 246, 0.18) !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
}

/* Textareas & Inputs */
.gradio-container textarea,
.gradio-container input[type="text"] {
    background: #161c36 !important;
    background-color: #161c36 !important;
    border: 1.5px solid rgba(139, 92, 246, 0.25) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14.5px !important;
    line-height: 1.65 !important;
    padding: 14px 16px !important;
    transition: all 0.2s ease !important;
}

.gradio-container textarea:focus,
.gradio-container input[type="text"]:focus {
    background: #19203d !important;
    background-color: #19203d !important;
    border-color: #9d7cfc !important;
    box-shadow: 0 0 0 4px rgba(157, 124, 252, 0.2) !important;
    outline: none !important;
}

.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
    color: #64748b !important;
}

/* Textbox Labels */
.gradio-container label span {
    color: #a5b4fc !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    margin-bottom: 6px !important;
    display: inline-block !important;
}

/* Primary Button */
button.primary-btn {
    background: linear-gradient(135deg, #7c3aed 0%, #6366f1 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    min-height: 50px !important;
    letter-spacing: 0.4px !important;
    box-shadow: 0 8px 24px rgba(124, 58, 237, 0.35) !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

button.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 30px rgba(124, 58, 237, 0.5) !important;
    filter: brightness(1.1) !important;
}

/* Secondary Button */
button.secondary-btn {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    color: #94a3b8 !important;
    font-size: 14.5px !important;
    font-weight: 600 !important;
    min-height: 50px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

button.secondary-btn:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    color: #ffffff !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
}

/* Preset / Sample Buttons */
button.preset-btn {
    background: rgba(139, 92, 246, 0.12) !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    border-radius: 20px !important;
    color: #c4b5fd !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 7px 14px !important;
    min-height: 34px !important;
    transition: all 0.2s ease !important;
}

button.preset-btn:hover {
    background: rgba(139, 92, 246, 0.25) !important;
    color: #ffffff !important;
    border-color: #8b5cf6 !important;
    transform: translateY(-1px) !important;
}

/* --- Hero Banner --- */
.hero-banner {
    background:
        radial-gradient(ellipse at 85% 15%, rgba(124, 58, 237, 0.45) 0%, transparent 50%),
        radial-gradient(ellipse at 15% 85%, rgba(99, 102, 241, 0.3) 0%, transparent 45%),
        linear-gradient(135deg, #0f1225 0%, #171b35 50%, #1e1848 100%);
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 20px;
    padding: 38px 42px 32px;
    margin-bottom: 24px;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45);
    position: relative;
    overflow: hidden;
}

.hero-banner .tagline {
    color: #a78bfa;
    font-size: 11.5px;
    font-weight: 800;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
}

.hero-banner h1 {
    font-size: 34px;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 10px;
    letter-spacing: -0.8px;
    line-height: 1.2;
}

.hero-banner h1 span {
    background: linear-gradient(135deg, #c084fc 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-banner p {
    color: #94a3b8;
    font-size: 14.5px;
    margin: 0 0 18px;
    max-width: 680px;
    line-height: 1.7;
}

.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.hero-badge {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    color: #cbd5e1;
    font-size: 12px;
    font-weight: 600;
    padding: 6px 14px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.hero-badge .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #a78bfa;
    box-shadow: 0 0 8px #a78bfa;
}

/* --- Section Panel Titles --- */
.panel-header {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #ffffff;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.panel-header .icon {
    width: 32px;
    height: 32px;
    border-radius: 10px;
    background: rgba(139, 92, 246, 0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}

/* --- Results Dashboard --- */
.results-container {
    background: #111528;
    border: 1px solid rgba(139, 92, 246, 0.22);
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 14px 35px rgba(0, 0, 0, 0.35);
}

/* Preview / Idle State Card */
.idle-card {
    background: linear-gradient(135deg, rgba(17, 21, 40, 0.7) 0%, rgba(24, 28, 56, 0.7) 100%);
    border: 1.5px dashed rgba(139, 92, 246, 0.25);
    border-radius: 18px;
    padding: 48px 32px;
    text-align: center;
    color: #94a3b8;
}

.idle-card .idle-icon {
    width: 64px;
    height: 64px;
    margin: 0 auto 16px;
    border-radius: 50%;
    background: rgba(139, 92, 246, 0.15);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 30px;
    border: 1px solid rgba(139, 92, 246, 0.3);
    box-shadow: 0 0 25px rgba(139, 92, 246, 0.2);
}

.idle-card h3 {
    color: #ffffff;
    font-size: 18px;
    font-weight: 700;
    margin: 0 0 8px;
}

.idle-card p {
    font-size: 14px;
    color: #94a3b8;
    max-width: 440px;
    margin: 0 auto 22px;
    line-height: 1.7;
}

.idle-features {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    max-width: 460px;
    margin: 0 auto;
    text-align: left;
}

.idle-feature-item {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 12.5px;
    color: #cbd5e1;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Score Hero Banner */
.score-hero {
    background: linear-gradient(135deg, #161b36 0%, #1e1b48 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    position: relative;
    overflow: hidden;
}

.score-hero::after {
    content: '';
    position: absolute;
    right: -40px;
    top: -40px;
    width: 160px;
    height: 160px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.2), transparent 70%);
    pointer-events: none;
}

.score-hero-left {
    flex: 1;
}

.score-label {
    font-size: 11.5px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 6px;
}

.score-number-row {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 12px;
}

.score-main {
    font-size: 46px;
    font-weight: 900;
    color: #ffffff;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -1.5px;
}

.score-denom {
    font-size: 16px;
    color: #64748b;
    font-weight: 700;
}

.score-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 16px;
    border-radius: 30px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.3px;
}

.pill-strong {
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.35);
    color: #4ade80;
}
.pill-strong .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 8px #22c55e;
}

.pill-partial {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.35);
    color: #fbbf24;
}
.pill-partial .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #f59e0b;
    box-shadow: 0 0 8px #f59e0b;
}

.pill-weak {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.35);
    color: #f87171;
}
.pill-weak .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ef4444;
    box-shadow: 0 0 8px #ef4444;
}

/* SVG Radial Gauge */
.radial-gauge {
    position: relative;
    width: 104px;
    height: 104px;
    flex-shrink: 0;
}

.radial-gauge svg {
    transform: rotate(-90deg);
}

.radial-gauge .bg-circle {
    fill: none;
    stroke: rgba(255, 255, 255, 0.08);
    stroke-width: 9;
}

.radial-gauge .fill-circle {
    fill: none;
    stroke-width: 9;
    stroke-linecap: round;
    transition: stroke-dashoffset 1s cubic-bezier(0.4, 0, 0.2, 1);
}

.radial-gauge-center {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.radial-gauge-center .pct {
    font-size: 20px;
    font-weight: 800;
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}

.radial-gauge-center .sub {
    font-size: 10px;
    color: #94a3b8;
    font-weight: 600;
    margin-top: 2px;
}

/* Metrics Grid */
.metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
    margin-bottom: 20px;
}

.metric-card {
    background: #161c36;
    border: 1px solid rgba(139, 92, 246, 0.18);
    border-radius: 14px;
    padding: 16px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(139, 92, 246, 0.4);
}

.metric-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}

.metric-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #94a3b8;
}

.metric-icon {
    font-size: 15px;
}

.metric-value {
    font-size: 24px;
    font-weight: 800;
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
    margin-bottom: 8px;
}

.metric-bar-bg {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    height: 5px;
    overflow: hidden;
}

.metric-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #8b5cf6, #c084fc);
}

/* Feedback Section */
.feedback-box {
    background: rgba(139, 92, 246, 0.08);
    border-left: 4px solid #8b5cf6;
    border-radius: 0 14px 14px 0;
    padding: 18px 20px;
    color: #e2e8f0;
    font-size: 14px;
    line-height: 1.8;
    margin-bottom: 22px;
}

.feedback-box-header {
    color: #a78bfa;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Concepts Section */
.concepts-section {
    background: #161c36;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 18px 20px;
}

.concept-group-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 12px 0 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.concept-group-title:first-child {
    margin-top: 0;
}

.chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 12px;
}

.concept-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

.chip-matched {
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #4ade80;
}

.chip-missing {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #fbbf24;
}

.chip-none {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #64748b;
    font-style: italic;
}

/* Error Card */
.error-card {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 14px;
    padding: 20px 22px;
    color: #fca5a5;
    font-size: 14px;
    line-height: 1.6;
}

.error-card strong {
    color: #f87171;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    font-size: 15px;
}

/* Footer */
.app-footer {
    text-align: center;
    color: #475569;
    padding: 30px 15px 20px;
    font-size: 12.5px;
    line-height: 2;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    margin-top: 24px;
}

.app-footer strong {
    color: #64748b;
}

@media (max-width: 768px) {
    .metrics-grid {
        grid-template-columns: 1fr;
    }
    .score-hero {
        flex-direction: column;
        align-items: flex-start;
    }
    .hero-banner {
        padding: 24px 20px;
    }
    .hero-banner h1 {
        font-size: 26px;
    }
}
"""


# =====================================================
# DEFAULT IDLE / PREVIEW HTML
# =====================================================

DEFAULT_RESULTS_HTML = """
<div class="results-container">
    <div class="idle-card">
        <div class="idle-icon">📊</div>
        <h3>Evaluation Dashboard Ready</h3>
        <p>
            Enter your question, reference answer, and student response on the left,
            or click one of the preset templates to test immediately.
        </p>

        <div class="idle-features">
            <div class="idle-feature-item">
                <span>⚡</span>
                <span>Supervised ML Prediction</span>
            </div>
            <div class="idle-feature-item">
                <span>🧠</span>
                <span>Sentence Semantic Vectors</span>
            </div>
            <div class="idle-feature-item">
                <span>🔍</span>
                <span>TF-IDF Keyword Similarity</span>
            </div>
            <div class="idle-feature-item">
                <span>🎯</span>
                <span>Key Concept Coverage</span>
            </div>
        </div>
    </div>
</div>
"""


# =====================================================
# ANSWER EVALUATION LOGIC & HTML GENERATION
# =====================================================

def evaluate_and_format(question, reference_answer, student_answer):
    if not str(question).strip():
        return f"""
        <div class="results-container">
            <div class="error-card">
                <strong>⚠ Input Missing</strong>
                Please enter the examination question before running the evaluation.
            </div>
        </div>
        """

    if not str(reference_answer).strip():
        return f"""
        <div class="results-container">
            <div class="error-card">
                <strong>⚠ Input Missing</strong>
                Please enter the instructor's reference answer for comparison.
            </div>
        </div>
        """

    if not str(student_answer).strip():
        return f"""
        <div class="results-container">
            <div class="error-card">
                <strong>⚠ Input Missing</strong>
                Please enter the student's descriptive answer to evaluate.
            </div>
        </div>
        """

    try:
        res = evaluate_answer(question, reference_answer, student_answer)

        score = float(res["predicted_score"])
        model_score = float(res["model_score"])
        tfidf = float(res["tfidf_similarity"])
        semantic = float(res["semantic_similarity"])
        coverage = float(res["concept_coverage"])
        covered = res["covered_concepts"]
        missing = res["potentially_missing_concepts"]
        feedback = res["feedback"]

        score_pct = max(0.0, min(100.0, (score / 5.0) * 100))

        # Radial Gauge math (r=42, circumference = 2 * pi * 42 = 263.89)
        circ = 263.89
        dash_offset = circ * (1.0 - (score_pct / 100.0))

        if score >= 4.0:
            pill_class = "pill-strong"
            pill_text = "✦ Strong Answer · High Concept Alignment"
            stroke_color = "#22c55e"
        elif score >= 2.5:
            pill_class = "pill-partial"
            pill_text = "◑ Partially Complete · Needs Key Details"
            stroke_color = "#f59e0b"
        else:
            pill_class = "pill-weak"
            pill_text = "✕ Needs Improvement · Weak or Off-Topic"
            stroke_color = "#ef4444"

        # Format covered and missing concept tags
        if covered:
            covered_chips = "".join(
                f'<span class="concept-chip chip-matched">✓ {html.escape(w)}</span>'
                for w in covered
            )
        else:
            covered_chips = '<span class="concept-chip chip-none">No key concepts matched</span>'

        if missing:
            missing_chips = "".join(
                f'<span class="concept-chip chip-missing">⚠ {html.escape(w)}</span>'
                for w in missing
            )
        else:
            missing_chips = '<span class="concept-chip chip-none">All key instructor concepts addressed!</span>'

        # Format feedback text
        formatted_feedback = html.escape(feedback).replace("\n\n", "<br><br>").replace("\n", " ")

        tfidf_pct = int(max(0, min(100, tfidf * 100)))
        sem_pct = int(max(0, min(100, semantic * 100)))
        cov_pct = int(max(0, min(100, coverage)))

        results_html = f"""
        <div class="results-container">
            <!-- Hero Score Card -->
            <div class="score-hero">
                <div class="score-hero-left">
                    <div class="score-label">Predicted Overall Assessment</div>
                    <div class="score-number-row">
                        <span class="score-main">{score:.2f}</span>
                        <span class="score-denom">/ 5.00</span>
                    </div>
                    <div class="score-status-pill {pill_class}">
                        <span class="dot"></span>
                        <span>{pill_text}</span>
                    </div>
                </div>

                <div class="radial-gauge">
                    <svg width="104" height="104" viewBox="0 0 104 104">
                        <circle class="bg-circle" cx="52" cy="52" r="42"/>
                        <circle class="fill-circle" cx="52" cy="52" r="42"
                                stroke="{stroke_color}"
                                stroke-dasharray="{circ:.2f}"
                                stroke-dashoffset="{dash_offset:.2f}"/>
                    </svg>
                    <div class="radial-gauge-center">
                        <div class="pct">{score_pct:.0f}%</div>
                        <div class="sub">MATCH</div>
                    </div>
                </div>
            </div>

            <!-- 3-Metrics Grid -->
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Semantic Match</span>
                        <span class="metric-icon">🧠</span>
                    </div>
                    <div class="metric-value">{semantic:.3f}</div>
                    <div class="metric-bar-bg">
                        <div class="metric-bar-fill" style="width: {sem_pct}%;"></div>
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">TF-IDF Overlap</span>
                        <span class="metric-icon">📊</span>
                    </div>
                    <div class="metric-value">{tfidf:.3f}</div>
                    <div class="metric-bar-bg">
                        <div class="metric-bar-fill" style="width: {tfidf_pct}%;"></div>
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-header">
                        <span class="metric-title">Concept Coverage</span>
                        <span class="metric-icon">🎯</span>
                    </div>
                    <div class="metric-value">{coverage:.1f}%</div>
                    <div class="metric-bar-bg">
                        <div class="metric-bar-fill" style="width: {cov_pct}%;"></div>
                    </div>
                </div>
            </div>

            <!-- Intelligent Diagnostic Feedback -->
            <div class="feedback-box">
                <div class="feedback-box-header">
                    <span>💡</span>
                    <span>AI Diagnostic Feedback</span>
                </div>
                {formatted_feedback}
            </div>

            <!-- Concepts Breakdown -->
            <div class="concepts-section">
                <div class="concept-group-title">
                    <span>✓</span>
                    <span>Covered Concepts in Student Answer</span>
                </div>
                <div class="chip-container">
                    {covered_chips}
                </div>

                <div class="concept-group-title" style="margin-top: 14px;">
                    <span>⚠</span>
                    <span>Missing Reference Concepts to Review</span>
                </div>
                <div class="chip-container">
                    {missing_chips}
                </div>
            </div>
        </div>
        """
        return results_html

    except Exception as err:
        err_msg = html.escape(str(err))
        return f"""
        <div class="results-container">
            <div class="error-card">
                <strong>⚠ Evaluation Error Encountered</strong>
                {err_msg}
            </div>
        </div>
        """


# =====================================================
# SAMPLE DATA PRESETS
# =====================================================

PRESET_1 = (
    "What is the role of a prototype program in problem solving?",
    "To simulate the behaviour of portions of the desired software product.",
    "A prototype simulates portions of the desired software product before its final development."
)

PRESET_2 = (
    "What is the role of a prototype program in problem solving?",
    "To simulate the behaviour of portions of the desired software product.",
    "A prototype is an early version of a software program."
)

PRESET_3 = (
    "What is the role of a prototype program in problem solving?",
    "To simulate the behaviour of portions of the desired software product.",
    "The weather is sunny today and I enjoy playing cricket with friends."
)


# =====================================================
# MAIN APPLICATION INTERFACE
# =====================================================

with gr.Blocks(
    title="AI-Powered Descriptive Answer Evaluation",
    fill_width=True
) as demo:

    # --- Header Banner ---
    gr.HTML("""
    <div class="hero-banner">
        <div class="tagline">✦ Intelligent Educational Technology</div>
        <h1>🎓 AI-Powered <span>Descriptive Answer Evaluation</span></h1>
        <p>
            Automated scoring and granular semantic analysis using Sentence Transformers,
            TF-IDF keyword mapping, and Supervised Machine Learning.
        </p>
        <div class="hero-badges">
            <span class="hero-badge"><span class="dot"></span> Semantic Embedding Analysis</span>
            <span class="hero-badge"><span class="dot"></span> ML Regression Scoring</span>
            <span class="hero-badge"><span class="dot"></span> Keyword Overlap (TF-IDF)</span>
            <span class="hero-badge"><span class="dot"></span> Automated Diagnostic Feedback</span>
        </div>
    </div>
    """)

    with gr.Row(equal_height=False):
        # --- Left Column: Inputs ---
        with gr.Column(scale=5):
            gr.HTML("""
            <div class="panel-header">
                <div class="icon">📝</div>
                <span>Assessment Details</span>
            </div>
            """)

            # Quick Presets
            gr.HTML("""
            <div style="margin-bottom: 12px; font-size: 12px; color: #94a3b8; font-weight: 600;">
                ⚡ QUICK TEST PRESETS:
            </div>
            """)
            with gr.Row():
                btn_preset1 = gr.Button("🟢 Strong Answer", elem_classes=["preset-btn"])
                btn_preset2 = gr.Button("🟡 Partial Answer", elem_classes=["preset-btn"])
                btn_preset3 = gr.Button("🔴 Off-Topic", elem_classes=["preset-btn"])

            question_input = gr.Textbox(
                label="Examination Question",
                placeholder="Enter the question asked in the exam...",
                lines=2,
            )

            reference_input = gr.Textbox(
                label="Instructor Reference Answer (Expected Concept)",
                placeholder="Enter the model / expected answer for evaluation reference...",
                lines=4,
            )

            student_input = gr.Textbox(
                label="Student's Descriptive Response",
                placeholder="Enter the descriptive response written by the student...",
                lines=7,
            )

            with gr.Row():
                evaluate_btn = gr.Button(
                    "✨ Evaluate Answer",
                    variant="primary",
                    elem_classes=["primary-btn"],
                    scale=3
                )
                clear_btn = gr.Button(
                    "⟳ Clear",
                    elem_classes=["secondary-btn"],
                    scale=1
                )

        # --- Right Column: Results Dashboard ---
        with gr.Column(scale=5):
            gr.HTML("""
            <div class="panel-header">
                <div class="icon">📊</div>
                <span>Evaluation Report</span>
            </div>
            """)

            results_panel = gr.HTML(value=DEFAULT_RESULTS_HTML)

    # --- Wire Button Actions ---
    evaluate_btn.click(
        fn=evaluate_and_format,
        inputs=[question_input, reference_input, student_input],
        outputs=[results_panel],
        show_progress="full"
    )

    clear_btn.click(
        fn=lambda: ("", "", "", DEFAULT_RESULTS_HTML),
        inputs=[],
        outputs=[question_input, reference_input, student_input, results_panel]
    )

    btn_preset1.click(
        fn=lambda: PRESET_1,
        inputs=[],
        outputs=[question_input, reference_input, student_input]
    )

    btn_preset2.click(
        fn=lambda: PRESET_2,
        inputs=[],
        outputs=[question_input, reference_input, student_input]
    )

    btn_preset3.click(
        fn=lambda: PRESET_3,
        inputs=[],
        outputs=[question_input, reference_input, student_input]
    )

    # --- Footer ---
    gr.HTML("""
    <div class="app-footer">
        <strong>AI-Powered Descriptive Answer Evaluation &amp; Feedback System</strong><br>
        Built with Python &nbsp;·&nbsp; Scikit-learn &nbsp;·&nbsp; Sentence Transformers &nbsp;·&nbsp; Gradio<br>
        Supervised Machine Learning Academic Project &nbsp;·&nbsp; © 2026 AI Assessment Platform
    </div>
    """)


# =====================================================
# LAUNCH APPLICATION
# =====================================================

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="violet",
            neutral_hue="slate"
        ),
        head='<script>document.documentElement.classList.add("dark");</script>'
    )
