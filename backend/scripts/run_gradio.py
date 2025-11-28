import os
import gradio as gr
import pandas as pd

# === BookRS imports (DB-backed models and utilities) ===
from backend.ml.recommender_semantic import SemanticRecommender
from backend.ml.recommender_hybrid import HybridRecommender
from backend.ml.recommender_popularity import PopularityRecommender
from backend.core.db_utils import load_books, count_records
from backend.core.config import EMB_PATH, EMB_META, ALS_USER_FACTORS, ALS_ITEM_FACTORS, POPULARITY_PATH
CUSTOM_CSS = """
<style>
:root { --radius: 10px; }
.gradio-container {
    font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto,
                 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif;
}
hr {
    border: none;
    border-top: 1px solid #eee;
    margin: 10px 0;
}
/* you can uncomment and adjust if needed:
.gr-button {
    height: 42px;
    font-weight: 600;
}
textarea {
    font-size: 16px !important;
    height: 42px !important;
}
*/
</style>
"""

# -------------------------------------------------------------------
# One-time model loads
# -------------------------------------------------------------------
print("[INFO] Loading Semantic / Hybrid / Popularity models ...")
semantic_model = SemanticRecommender()
hybrid_model   = HybridRecommender()
pop_model      = PopularityRecommender()
print("[OK] All models ready.")

# Cover image map (optional)
try:
    _books = load_books(columns=["book_id", "title", "authors", "image_url"])
    cover_map = _books.rename(columns={"book_id": "book_id"})[["book_id", "image_url"]]
except Exception as e:
    print("[WARN] Could not load image URLs:", e)
    cover_map = pd.DataFrame(columns=["book_id", "image_url"])

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
PLACEHOLDER = "https://cdn-icons-png.flaticon.com/512/29/29302.png"

def _attach_covers(df: pd.DataFrame) -> pd.DataFrame:
    if "book_id" not in df.columns or cover_map.empty:
        df["image_url"] = PLACEHOLDER
        return df
    out = df.merge(cover_map, on="book_id", how="left")
    out["image_url"] = out["image_url"].fillna(PLACEHOLDER)
    return out

def _cards_html(df: pd.DataFrame, score_label: str = "Score") -> str:
    # Build a responsive, clean card list
    if df is None or len(df) == 0:
        return "<div style='color:#666;padding:8px;'>No results.</div>"
    rows = []
    for _, r in df.iterrows():
        img = r.get("image_url", PLACEHOLDER)
        title = str(r.get("title", "")).strip()
        authors = str(r.get("authors", "")).strip()
        score = r.get("semantic_score", r.get("hybrid_score", r.get("tfidf_score", r.get("popularity_score", ""))))
        score_str = f"{score_label}: {round(float(score), 4)}" if score != "" and pd.notna(score) else ""
        row = f"""
        <div style="display:flex;gap:14px;align-items:center;padding:10px 12px;border-bottom:1px solid #eee;">
          <img src="{img}" alt="cover" width="72" height="96"
               style="border-radius:8px;object-fit:cover;background:#f7f7f7"
               onerror="this.src='{PLACEHOLDER}';">
          <div style="display:flex;flex-direction:column;">
            <div style="font-weight:600;font-size:1.05rem;line-height:1.3">{title}</div>
            <div style="color:#555">{authors}</div>
            <div style="color:#888;margin-top:4px">{score_str}</div>
          </div>
        </div>
        """
        rows.append(row)
    return "<div style='max-width:820px;margin:0 auto;border:1px solid #eee;border-radius:10px;overflow:hidden;'>" + "".join(rows) + "</div>"

# -------------------------------------------------------------------
# Tab functions
# -------------------------------------------------------------------
def search_handler(query: str, mode: str, user_id: int, k: int):
    if not query or not str(query).strip():
        return "<div style='color:#666;padding:8px;'>Please enter a query.</div>"
    if mode == "Relevant (Semantic)":
        df = semantic_model.recommend(query, top_k=k)
        df = _attach_covers(df)
        return _cards_html(df, score_label="Semantic")
    # For You (Hybrid)
    df = hybrid_model.recommend(query, user_id=int(user_id or 0), top_k=k)
    df = _attach_covers(df)
    return _cards_html(df, score_label="Hybrid")

def home_feed_handler(user_id: int, k: int):
    uid = int(user_id or 0)
    if uid <= 0:
        df = pop_model.recommend(top_k=k)
        df = _attach_covers(df)
        return "Guest (Popular Now)", _cards_html(df, score_label="Popularity")
    # Personalized: call hybrid with a neutral query.
    # (In a v2, you can re-rank popular candidates purely by CF.)
    df = hybrid_model.recommend(query="recommended", user_id=uid, top_k=k)
    df = _attach_covers(df)
    return f"User {uid} (Personalized)", _cards_html(df, score_label="Hybrid")

def popular_handler(k: int):
    df = pop_model.recommend(top_k=k)
    df = _attach_covers(df)
    return _cards_html(df, score_label="Popularity")

def model_info_handler():
    counts = count_records()
    # Artifacts present?
    art = {
        "Embeddings (pt)": os.path.exists(EMB_PATH),
        "Embeddings meta (parquet)": os.path.exists(EMB_META),
        "ALS user factors (npz)": os.path.exists(ALS_USER_FACTORS),
        "ALS item factors (npz)": os.path.exists(ALS_ITEM_FACTORS),
        "Popularity (parquet)": os.path.exists(POPULARITY_PATH),
    }
    # Render simple HTML summary
    rows_counts = "".join([
        f"<tr><td style='padding:6px 10px;border-bottom:1px solid #eee;'>{k}</td>"
        f"<td style='padding:6px 10px;border-bottom:1px solid #eee;text-align:right'>{v:,}</td></tr>"
        for k, v in counts.items()
    ])
    rows_art = "".join([
        f"<tr><td style='padding:6px 10px;border-bottom:1px solid #eee;'>{k}</td>"
        f"<td style='padding:6px 10px;border-bottom:1px solid #eee;'>{'✅' if v else '❌'}</td></tr>"
        for k, v in art.items()
    ])
    html = f"""
    <div style="max-width:820px;margin:0 auto;">
      <h3>Dataset & Table Counts</h3>
      <table style="width:100%;border-collapse:collapse;margin-bottom:16px;">
        <thead><tr><th style='text-align:left;padding:6px 10px;'>Table</th><th style='text-align:right;padding:6px 10px;'>Rows</th></tr></thead>
        <tbody>{rows_counts}</tbody>
      </table>
      <h3>Artifacts Present</h3>
      <table style="width:100%;border-collapse:collapse;">
        <thead><tr><th style='text-align:left;padding:6px 10px;'>File</th><th style='text-align:left;padding:6px 10px;'>Status</th></tr></thead>
        <tbody>{rows_art}</tbody>
      </table>
      <p style="color:#666">Semantic model: <code>all-MiniLM-L6-v2</code> · Hybrid weights (example): <code>0.7 × semantic + 0.3 × CF</code></p>
    </div>
    """
    return html


# -------------------------------------------------------------------
# Gradio UI
# -------------------------------------------------------------------
with gr.Blocks(title="BookRS — AI-Powered Book Recommendation System") as demo:
    # Inject custom CSS for Gradio 4+ (no css= param anymore)
    gr.HTML(CUSTOM_CSS)

    gr.Markdown(
        "<h1 style='text-align:center; font-weight:700; font-family:Inter,system-ui; margin-bottom:0.5em;'>"
        "BookRS — AI-Powered Book Recommendation System</h1>",
        elem_id="bookrs-title"
    )

    gr.Markdown("Explore **Semantic Search**, **Hybrid (For You)**, and **Popular Books**. "
                "Use the user selector to simulate guest vs logged-in.")


    with gr.Accordion("👤 User / Login Simulation", open=True):
        user_id_global = gr.Number(value=0, label="User ID (0 = Guest, >0 = Logged-in)", interactive=True)

        # user_id_global = gr.Number(value=0, label="User ID (0 = Guest, >0 = Logged-in User)")

    with gr.Tab("Home Feed"):
        gr.Markdown("Guest → Popular Now · Logged-in → Personalized (Hybrid)")
        k_home = gr.Slider(5, 20, value=10, step=1, label="How many books?")
        mode_label = gr.Textbox(label="Mode", interactive=False)
        out_home = gr.HTML()
        gr.Button("Show Home Recommendations").click(
            fn=home_feed_handler,
            inputs=[user_id_global, k_home],
            outputs=[mode_label, out_home]
        )



        # auto-refresh when user ID changes
        user_id_global.change(
            fn=home_feed_handler,
            inputs=[user_id_global, k_home],
            outputs=[mode_label, out_home],
            show_progress=False,
            queue=False
        )


    with gr.Tab("Search"):
        gr.Markdown("Search by topic/title and choose **Relevant (Semantic)** or **For You (Hybrid)**.")

        # --- Row with input + search button side by side ---
        with gr.Row(equal_height=True):
            query = gr.Textbox(
                label="Enter a topic or title",
                placeholder="e.g., deep learning, fantasy, leadership …",
                scale=8

            )
            search_btn = gr.Button("Search", variant="primary", scale=1)

        # --- Second row: mode & number of results ---
        with gr.Row():
            mode = gr.Radio(
                choices=["Relevant (Semantic)", "For You (Hybrid)"],
                value="Relevant (Semantic)",
                label="Mode",
                scale=2
            )
            k_search = gr.Slider(3, 20, value=10, step=1, label="Results", scale=3)

        # --- Results output ---
        out_search = gr.HTML()

        # --- Event binding ---
        search_btn.click(
            fn=search_handler,
            inputs=[query, mode, user_id_global, k_search],
            outputs=[out_search]
        )

        # 🔹 Connect the Search button
        search_btn.click(
            fn=search_handler,
            inputs=[query, mode, user_id_global, k_search],
            outputs=[out_search]
        )

        # 🔹 Enable Enter key submission from the Textbox
        query.submit(
            fn=search_handler,
            inputs=[query, mode, user_id_global, k_search],
            outputs=[out_search]
        )


    with gr.Tab("Popular Books"):
        gr.Markdown("IMDb-style weighted popularity (average rating × rating count). Great for guests.")
        k_pop = gr.Slider(5, 30, value=12, step=1, label="How many books?")
        out_pop = gr.HTML()
        gr.Button("Show Popular").click(fn=popular_handler, inputs=[k_pop], outputs=[out_pop])

    with gr.Tab("Model Info"):
        info_html = gr.HTML()
        gr.Button("Refresh Info").click(fn=model_info_handler, inputs=[], outputs=[info_html])

# Entry point
if __name__ == "__main__":
    # Tip: if Gradio is behind a proxy/VPN, you can set server_name="0.0.0.0"
    demo.launch(server_name="0.0.0.0",server_port=8000)
