from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
import base64
import os
from datetime import datetime
from collections import Counter
from matplotlib.colors import LinearSegmentedColormap
import uvicorn

# Initialize the app
app = FastAPI(title="Automated Media Monitoring API")

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to encode matplotlib figures to Base64
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# Pydantic Models
class Article(BaseModel):
    author: str
    content_excerpt: str
    published_at: int
    sentiment_label: str
    title: str
    source_link: str
    keywords: list[str] = []

class JSONData(BaseModel):
    data: list[Article]

@app.post("/analyze_json")
async def analyze_json(payload: JSONData):
    raw_data = payload.data
    df = pd.DataFrame([article.dict() for article in raw_data])

    # Convert timestamp to datetime
    df["articleCreatedDate"] = df["published_at"].apply(lambda ts: datetime.utcfromtimestamp(ts))
    df = df.rename(columns={
        "author": "authorName",
        "sentiment_label": "sentimentHumanReadable",
    })

    # KPIs
    kpis = {
        "total_mentions": len(df),
        "positive": int((df["sentimentHumanReadable"] == "positive").sum()),
        "negative": int((df["sentimentHumanReadable"] == "negative").sum()),
        "neutral": int((df["sentimentHumanReadable"] == "neutral").sum()),
    }

    # Mentions over time
    df["Period"] = df["articleCreatedDate"].dt.date
    mentions_over_time = df["Period"].value_counts().sort_index()

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(mentions_over_time.index.astype(str), mentions_over_time.values, marker='o', linestyle='-', color="#023047")
    ax1.set_ylabel("Mentions")
    plt.xticks(rotation=45)

    # Add values above each 
    for x, y in zip(mentions_over_time.index.astype(str), mentions_over_time.values):
        ax1.text(x, y + 0.18, str(y), ha='center', fontsize=9)

    evolution_mentions_b64 = fig_to_base64(fig1)
    plt.close(fig1)

    # WordCloud + auto summary
    all_keywords = [kw.lower() for sublist in df["keywords"] if isinstance(sublist, list) for kw in sublist]
    summary_text = ""
    if all_keywords:
        keywords_text = " ".join(all_keywords)
        custom_cmap = LinearSegmentedColormap.from_list("custom_blue", ["#023047", "#023047"])
        wordcloud = WordCloud(width=800, height=400, background_color="white", colormap=custom_cmap).generate(keywords_text)

        fig_kw, ax_kw = plt.subplots(figsize=(10, 5))
        ax_kw.imshow(wordcloud, interpolation='bilinear')
        ax_kw.axis("off")
        keywords_freq_b64 = fig_to_base64(fig_kw)
        plt.close(fig_kw)

        # Auto summary based on top keywords
        counter = Counter(all_keywords)
        top_keywords = [kw for kw, _ in counter.most_common(6)]
        if top_keywords:
            summary_text = (
                f"The most frequent keywords in media coverage are "
                f"{', '.join(top_keywords[:-1])} and {top_keywords[-1]}. "
                f"This reflects the main topics discussed in the analyzed articles."
            )
    else:
        keywords_freq_b64 = ""

    # Sentiment by author
    author_sentiment = df.groupby(['authorName', 'sentimentHumanReadable']).size().unstack(fill_value=0)
    author_sentiment['Total'] = author_sentiment.sum(axis=1)
    top_authors_sentiment = author_sentiment.sort_values(by='Total', ascending=False).head(10).drop(columns='Total')
    top_authors_sentiment = top_authors_sentiment.iloc[::-1]

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    top_authors_sentiment.plot(kind='barh', stacked=True, ax=ax3, color="#023047")
    ax3.set_xlabel("Number of Articles")
    ax3.set_ylabel("Author")

    # Add values next to bars
    for i, (author, row) in enumerate(top_authors_sentiment.iterrows()):
        total_articles = row.sum()
        ax3.text(total_articles + 0.1, i, str(total_articles), va='center', fontsize=9)

    author_sentiment_b64 = fig_to_base64(fig3)
    plt.close(fig3)

    # Top authors table
    top_table = (
        df["authorName"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "count", "authorName": "Author"})
        .head(5)
        .to_html(index=False, border=1, classes="styled-table")
    )

    # Final HTML report 
    html_report = f"""<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 40px; max-width: 900px; margin: auto; background-color: white; }}
        h1, h2 {{ text-align: center; color: #023047; }}
        .centered-text {{ max-width: 800px; margin: 0 auto 40px; text-align: center; font-size: 16px; line-height: 1.6; }}
        .styled-table {{ border-collapse: collapse; margin: 25px auto; font-size: 16px; width: 80%; border: 1px solid #dddddd; }}
        .styled-table th, .styled-table td {{ padding: 10px 15px; text-align: left; border: 1px solid #dddddd; }}
        .styled-table thead th {{ background-color: white; font-weight: bold; }}
        .image-block {{ text-align: center; margin: 30px 0; }}
    </style>
</head>
<body>
    <div style="display: flex; justify-content: space-around; margin: 20px 0;">
        <div style="text-align: center;"><h3>{kpis['total_mentions']}</h3><p>Total Mentions</p></div>
        <div style="text-align: center;"><h3>{kpis['positive']}</h3><p>Positive</p></div>
        <div style="text-align: center;"><h3>{kpis['negative']}</h3><p>Negative</p></div>
        <div style="text-align: center;"><h3>{kpis['neutral']}</h3><p>Neutral</p></div>
    </div>
    <div class="image-block">
        <img src="data:image/png;base64,{evolution_mentions_b64}" width="700"/>
    </div>
    <div class="image-block">
        <img src="data:image/png;base64,{keywords_freq_b64}" width="600"/>
    </div>
    <div class="image-block">
        <img src="data:image/png;base64,{author_sentiment_b64}" width="700"/>
    </div>
    {top_table}
</body>
</html>
"""
    os.makedirs("static", exist_ok=True)
    with open("static/media_report.html", "w", encoding="utf-8") as f:
        f.write(html_report)

    return {
        "kpis": kpis,
        "html_report": html_report,
        "summary_text": summary_text
    }

@app.get("/report")
def get_report():
    return FileResponse("static/media_report.html", media_type="text/html")

# Auto-start for Railway
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
