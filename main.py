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

# Initialisation de l'app
app = FastAPI(title="API Analyse Veille Médiatique")

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fonction d'encodage des graphiques
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# Modèles Pydantic
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

@app.post("/analyser_json")
async def analyser_json(payload: JSONData):
    raw_data = payload.data
    df = pd.DataFrame([article.dict() for article in raw_data])

    df["articleCreatedDate"] = df["published_at"].apply(lambda ts: datetime.utcfromtimestamp(ts))
    df = df.rename(columns={
        "author": "authorName",
        "sentiment_label": "sentimentHumanReadable",
    })

    kpis = {
        "total_mentions": len(df),
        "positive": int((df["sentimentHumanReadable"] == "positive").sum()),
        "negative": int((df["sentimentHumanReadable"] == "negative").sum()),
        "neutral": int((df["sentimentHumanReadable"] == "neutral").sum()),
    }

    # Évolution des mentions
    df["Period"] = df["articleCreatedDate"].dt.date
    mentions_over_time = df["Period"].value_counts().sort_index()
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(mentions_over_time.index.astype(str), mentions_over_time.values, marker='o', linestyle='-', color="#023047")
    ax1.set_title("Évolution des mentions par jour")
    ax1.set_ylabel("Mentions")
    plt.xticks(rotation=45)
    evolution_mentions_b64 = fig_to_base64(fig1)
    plt.close(fig1)

    # WordCloud des mots-clés + résumé automatique
    all_keywords = [kw.lower() for sublist in df["keywords"] if isinstance(sublist, list) for kw in sublist]
    summary_text = ""
    if all_keywords:
        # Génération du WordCloud
        keywords_text = " ".join(all_keywords)
        custom_cmap = LinearSegmentedColormap.from_list("custom_blue", ["#023047", "#023047"])
        wordcloud = WordCloud(width=800, height=400, background_color="white", colormap=custom_cmap).generate(keywords_text)
        fig_kw, ax_kw = plt.subplots(figsize=(10, 5))
        ax_kw.imshow(wordcloud, interpolation='bilinear')
        ax_kw.axis("off")
        ax_kw.set_title("Mots-clés les plus fréquents", fontsize=16)
        keywords_freq_b64 = fig_to_base64(fig_kw)
        plt.close(fig_kw)

        # Résumé basé sur les mots-clés dominants (top 6)
        counter = Counter(all_keywords)
        top_keywords = [kw for kw, _ in counter.most_common(6)]
        if top_keywords:
            summary_text = (
                f"Les mots-clés les plus récurrents dans la couverture médiatique sont "
                f"{', '.join(top_keywords[:-1])} et {top_keywords[-1]}. "
                f"Cela reflète les thématiques principales abordées dans les articles analysés."
            )
    else:
        keywords_freq_b64 = ""

    # Sentiments par auteur (graphique horizontal)
    author_sentiment = df.groupby(['authorName', 'sentimentHumanReadable']).size().unstack(fill_value=0)
    author_sentiment['Total'] = author_sentiment.sum(axis=1)
    top_authors_sentiment = author_sentiment.sort_values(by='Total', ascending=False).head(10).drop(columns='Total')
    top_authors_sentiment = top_authors_sentiment.iloc[::-1]

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    top_authors_sentiment.plot(kind='barh', stacked=True, ax=ax3, color="#023047")
    ax3.set_xlabel("Nombre d'articles")
    ax3.set_ylabel("Auteur")
    ax3.set_title("Répartition des sentiments par auteur")
    sentiments_auteurs_b64 = fig_to_base64(fig3)
    plt.close(fig3)

    # Tableau top auteurs
    top_table = (
        df["authorName"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "count", "authorName": "Auteur"})
        .head(5)
        .to_html(index=False, border=1, classes="styled-table")
    )

    # HTML final
    html_report = f"""<!DOCTYPE html>
<html lang='fr'>
<head>
    <meta charset='UTF-8'>
    <title>📊 Rapport d'Analyse Automatisée de Veille Médiatique</title>
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
    <h1>📊 Rapport d'Analyse Automatisée de Veille Médiatique</h1>
    <div class="centered-text">
        <p>
            Ce rapport fournit une analyse approfondie des articles collectés depuis la plateforme Lumenfeed. 
            Il vise à offrir une vision claire et synthétique de la couverture médiatique d’un sujet donné, 
            en mettant en évidence les volumes de publication, les auteurs les plus actifs, et les principaux mots-clés abordés. 
        </p>
    </div>
    <h2>Indicateurs Clés</h2>
    <div style="display: flex; justify-content: space-around; margin: 20px 0;">
        <div style="text-align: center;"><h3>{kpis['total_mentions']}</h3><p>Mentions totales</p></div>
        <div style="text-align: center;"><h3>{kpis['positive']}</h3><p>Positives</p></div>
        <div style="text-align: center;"><h3>{kpis['negative']}</h3><p>Négatives</p></div>
        <div style="text-align: center;"><h3>{kpis['neutral']}</h3><p>Neutres</p></div>
    </div>
    <div class="image-block">
        <h2>Évolution des mentions</h2>
        <img src="data:image/png;base64,{evolution_mentions_b64}" width="700"/>
    </div>
    <div class="image-block">
        <h2>Mots-clés les plus fréquents</h2>
        <img src="data:image/png;base64,{keywords_freq_b64}" width="600"/>
    </div>
    <h2>Résumé automatique</h2>
    <div class="centered-text">
        <p>{summary_text}</p>
    </div>
    <div class="image-block">
        <h2>Répartition des sentiments par auteur</h2>
        <img src="data:image/png;base64,{sentiments_auteurs_b64}" width="700"/>
    </div>
    <h2>Top 5 Auteurs les plus actifs</h2>
    {top_table}
</body>
</html>
"""
    os.makedirs("static", exist_ok=True)
    with open("static/rapport_veille.html", "w", encoding="utf-8") as f:
        f.write(html_report)

    return {
        "kpis": kpis,
        "html_report": html_report
    }

@app.get("/rapport")
def get_rapport():
    return FileResponse("static/rapport_veille.html", media_type="text/html")
