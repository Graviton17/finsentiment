import streamlit as st
from model import load_finbert
from data_fetcher import fetch_company_news, fetch_news_yfinance
from utils import run_inference, aggregate_sentiment
from pdf_extractor import extract_text_from_pdf, pdf_meta
import ui_components as ui

st.set_page_config(page_title="FinSentiment Analyzer", page_icon="📈", layout="wide")

@st.cache_resource(show_spinner="Loading FinBERT model (first run only)…")
def get_model():
    return load_finbert()

pipe = get_model()

st.title("📈 Financial Sentiment Analyzer")
st.markdown("Powered by **ProsusAI/FinBERT** — domain-adapted BERT for financial news.")
st.divider()

# Delegate sidebar to UI module
input_mode, ticker, days_back, data_source, uploaded_pdf, analyze_btn = ui.render_sidebar()

if uploaded_pdf:
    meta = pdf_meta(uploaded_pdf)
    uploaded_pdf.seek(0)
    st.sidebar.success(f"**{meta['title']}**")
    st.sidebar.caption(f"Author: {meta['author']}  •  {meta['pages']} pages")

if analyze_btn:
    articles = []
    label = ""
    
    if input_mode == "📄 Upload PDF":
        if not uploaded_pdf:
            st.warning("Please upload a PDF file first.")
            st.stop()
        with st.spinner("Extracting text from PDF…"):
            uploaded_pdf.seek(0)
            articles = extract_text_from_pdf(uploaded_pdf)
        if not articles:
            st.error("Could not extract text from this PDF. It may be scanned/image-only.")
            st.stop()
        label = uploaded_pdf.name
        st.success(f"Extracted **{len(articles)} pages** of text. Running FinBERT…")

    else:
        if not ticker:
            st.warning("Please enter or select a ticker symbol.")
            st.stop()
            
        with st.spinner(f"Fetching news for **{ticker}**…"):
            # ADD THIS LINE to satisfy the type checker
            assert days_back is not None 
            
            articles = (
                fetch_company_news(ticker, days_back)
                if data_source == "Finnhub"
                else fetch_news_yfinance(ticker)
            )

    # Shared Inference Flow
    with st.spinner("Analyzing sentiment…"):
        results = run_inference(pipe, articles)
        summary = aggregate_sentiment(results)

    # Delegate visualizations to UI module
    is_pdf = input_mode == "📄 Upload PDF"
    ui.render_kpi_cards(summary, label, is_pdf)
    ui.render_charts(summary)
    ui.render_timeline(results)
    ui.render_results_table(results, label)

else:
    st.info("👈 Choose an input source in the sidebar and click **Analyze**.")