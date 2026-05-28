import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Financial Sentiment Analyzer", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main { background-color: #1a1f2e; }
    .stApp { background-color: #1a1f2e; color: #ffffff; }
    h1, h2, h3 { color: #F0A500; }
    .stTextArea textarea { background-color: #2a2f3e; color: #ffffff; }
    .metric-card {
        background-color: #2a2f3e;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #F0A500;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Financial Sentiment Analyzer")
st.markdown("**Powered by FinBERT — ProsusAI/finbert**")

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="ProsusAI/finbert")

with st.spinner("Loading FinBERT model..."):
    finbert = load_model()

tab1, tab2 = st.tabs(["🔍 Analyze Text", "📂 Batch Analysis"])

with tab1:
    st.subheader("Single Text Analysis")
    text = st.text_area("Enter financial text:", height=150, placeholder="e.g. Apple reported record profits beating analyst expectations.")
    
    if st.button("Analyze", type="primary"):
        if text.strip():
            with st.spinner("Analyzing..."):
                result = finbert(text[:512])
                top = max(result, key=lambda x: x["score"])
                label = top["label"].upper()
                score = round(top["score"] * 100, 2)

                col1, col2 = st.columns(2)
                with col1:
                    color = "#00C853" if label == "POSITIVE" else "#D32F2F" if label == "NEGATIVE" else "#F0A500"
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3 style='color:{color}'>{label}</h3>
                        <h2 style='color:#ffffff'>{score}% confidence</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    fig = px.bar(
                        x=[r["label"] for r in result],
                        y=[round(r["score"]*100, 2) for r in result],
                        color=[r["label"] for r in result],
                        color_discrete_map={"positive": "#00C853", "negative": "#D32F2F", "neutral": "#F0A500"},
                        labels={"x": "Sentiment", "y": "Confidence %"}
                    )
                    fig.update_layout(
                        paper_bgcolor="#2a2f3e",
                        plot_bgcolor="#2a2f3e",
                        font_color="#ffffff",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Batch CSV Analysis")
    uploaded = st.file_uploader("Upload CSV (must have a 'text' column)", type="csv")
    
    if uploaded:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("CSV must have a 'text' column.")
        else:
            df = df.head(100)
            st.info(f"Analyzing {len(df)} rows...")
            
            results = []
            progress = st.progress(0)
            for i, row in df.iterrows():
                res = finbert(str(row["text"])[:512])
                top = max(res, key=lambda x: x["score"])
                results.append({"text": row["text"], "sentiment": top["label"], "confidence": round(top["score"]*100, 2)})
                progress.progress((i+1)/len(df))
            
            result_df = pd.DataFrame(results)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(result_df, names="sentiment", title="Sentiment Distribution",
                            color="sentiment",
                            color_discrete_map={"positive": "#00C853", "negative": "#D32F2F", "neutral": "#F0A500"})
                fig.update_layout(paper_bgcolor="#2a2f3e", plot_bgcolor="#2a2f3e", font_color="#ffffff")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.dataframe(result_df, use_container_width=True)
            
            csv = result_df.to_csv(index=False)
            st.download_button("Download Results", csv, "sentiment_results.csv", "text/csv")