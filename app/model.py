# model.py
from transformers import pipeline

def load_finbert():
    """
    Load the ProsusAI/finbert sentiment pipeline.
    This is called once and cached by Streamlit.
    The model is ~400 MB; first run will download it automatically.
    """
    return pipeline(
        task="text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        top_k=None,          # Return scores for ALL labels (positive/negative/neutral)
        device=-1,           # -1 = CPU; change to 0 for GPU
    )