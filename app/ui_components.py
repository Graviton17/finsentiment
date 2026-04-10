import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def render_sidebar():
    """Renders the sidebar and returns user inputs."""
    with st.sidebar:
        st.header("⚙️ Settings")
        input_mode = st.radio("Input source", ["📈 Stock ticker", "📄 Upload PDF"], horizontal=True)
        st.divider()

        ticker, days_back, data_source, uploaded_pdf = None, None, None, None

        if input_mode == "📈 Stock ticker":
            TICKERS = {
                "AAPL  — Apple": "AAPL", "MSFT  — Microsoft": "MSFT", 
                "GOOGL — Alphabet": "GOOGL", "AMZN  — Amazon": "AMZN",
                "CUSTOM — Enter manually": "CUSTOM",
            }
            selected_label = st.selectbox("Stock ticker", options=list(TICKERS.keys()))
            selected_value = TICKERS[selected_label]

            ticker = st.text_input("Enter ticker symbol", placeholder="e.g. UBER").upper().strip() if selected_value == "CUSTOM" else selected_value
            if ticker != "CUSTOM": st.caption(f"Selected: `{ticker}`")

            days_back = st.slider("News lookback (days)", 1, 30, 7)
            data_source = st.radio("Data source", ["Finnhub", "yfinance (no key needed)"])
        else:
            uploaded_pdf = st.file_uploader("Upload a financial PDF", type=["pdf"])

        st.divider()
        analyze_btn = st.button("🔍 Analyze", width="stretch", type="primary")
        
        return input_mode, ticker, days_back, data_source, uploaded_pdf, analyze_btn

def render_kpi_cards(summary, label, is_pdf_mode):
    st.subheader(f"Sentiment summary — {label}")
    col1, col2, col3, col4 = st.columns(4)

    sentiment_emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
    overall = summary.get("overall_sentiment", "neutral")
    composite = summary.get("composite_score", 0.0)

    col1.metric("Overall sentiment", f"{sentiment_emoji.get(overall, '⚪')} {overall.capitalize()}")
    col2.metric("Composite score", f"{composite:+.3f}")
    col3.metric("Units analyzed", f"{summary.get('total_articles', 0)} {'pages' if is_pdf_mode else 'articles'}")
    col4.metric("Bullish / Bearish", f"{summary.get('positive_count', 0)} / {summary.get('negative_count', 0)}")
    st.divider()

def render_charts(summary):
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Volume breakdown")
        pie_df = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Count": [summary.get("positive_count", 0), summary.get("negative_count", 0), summary.get("neutral_count", 0)],
        })
        fig_pie = px.pie(pie_df, names="Sentiment", values="Count", color="Sentiment",
                        color_discrete_map={"Positive":"#22c55e","Negative":"#ef4444","Neutral":"#94a3b8"}, hole=0.45)
        fig_pie.update_layout(showlegend=True, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_pie, width="stretch")

    with chart_col2:
        st.subheader("Average label confidence")
        bar_df = pd.DataFrame({
            "Label": ["Positive", "Negative", "Neutral"],
            "Avg score": [summary.get("avg_positive", 0), summary.get("avg_negative", 0), summary.get("avg_neutral", 0)],
        })
        bar_df["label_text"] = bar_df["Avg score"].map(lambda v: f"{v:.3f}")
        fig_bar = px.bar(bar_df, x="Label", y="Avg score", color="Label",
                        color_discrete_map={"Positive":"#22c55e","Negative":"#ef4444","Neutral":"#94a3b8"}, text="label_text")
        fig_bar.update_layout(showlegend=False, yaxis_range=[0,1], margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_bar, width="stretch")

def render_timeline(results):
    st.subheader("Sentiment timeline")
    timeline_df = pd.DataFrame(results)
    if timeline_df.empty:
        return
        
    timeline_df["date"] = pd.to_datetime(timeline_df["datetime"]).dt.date
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    timeline_df["score_numeric"] = timeline_df["sentiment"].map(sentiment_map)
    
    daily = timeline_df.groupby("date").agg(avg_score=("score_numeric","mean"), article_count=("headline","count")).reset_index().sort_values("date")
    
    fig_tl = go.Figure()
    fig_tl.add_bar(x=daily["date"], y=daily["article_count"], name="Count", marker_color="#cbd5e1", yaxis="y2", opacity=0.5)
    fig_tl.add_scatter(x=daily["date"], y=daily["avg_score"], mode="lines+markers", name="Avg sentiment",
                    line=dict(color="#6366f1", width=2), marker=dict(size=7))
    fig_tl.update_layout(yaxis=dict(title="Sentiment score", range=[-1.1,1.1], zeroline=True),
                        yaxis2=dict(title="Count", overlaying="y", side="right"),
                        legend=dict(orientation="h", y=1.1), margin=dict(t=20,b=20,l=40,r=40), hovermode="x unified")
    st.plotly_chart(fig_tl, width="stretch")

def render_results_table(results, label):
    st.subheader("Detailed results")
    display_df = pd.DataFrame(results)[
        ["datetime","source","sentiment","confidence",
        "positive_score","negative_score","neutral_score","headline","url"]
    ].sort_values("datetime", ascending=False)

    def highlight_sentiment(val):
        return {"positive": "background-color:#bbf7d0;color:#15803d",
                "negative": "background-color:#fecaca;color:#b91c1c",
                "neutral":  "background-color:#e2e8f0;color:#475569"}.get(val, "")

    st.dataframe(
        display_df.style.map(highlight_sentiment, subset=["sentiment"]),
        width="stretch",
        column_config={
            "url": st.column_config.LinkColumn("Link"),
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.3f"),
        },
        height=420,
    )
    
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name=f"{label}_sentiment.csv", mime="text/csv")