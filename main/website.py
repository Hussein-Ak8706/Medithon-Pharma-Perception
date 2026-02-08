import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud
import base64
from io import BytesIO
from collections import Counter

# =========================
# LOAD DATA
# =========================

# Load aggregated data
df_agg = pd.read_csv("Datasets/AnalyticData.csv")
df_agg.columns = [c.strip().replace(" ", "_").lower() for c in df_agg.columns]

# Load best/worst comments
df_comments = pd.read_csv("Datasets/DrugCommentSorted.csv")
df_comments.columns = [c.strip().replace(" ", "_").lower() for c in df_comments.columns]

# Load review counts
df_review_counts = pd.read_csv("Datasets/DrugReviewCount.csv")
# The CSV header is misleading - first column is drug name, second is count
# So we need to rename: first column (number_of_reviews) -> drug_name, second column (comment) -> number_of_reviews
df_review_counts.columns = ['drug_name', 'number_of_reviews']
# Convert drug_name to string to match the main dataframe
df_review_counts['drug_name'] = df_review_counts['drug_name'].astype(str)

# Merge
df = df_agg.merge(df_comments[['drug_name', 'most_positive_comment', 'most_negative_comment']], 
                  on='drug_name', how='left')
# Ensure drug_name is string type in main df before merging
df['drug_name'] = df['drug_name'].astype(str)
df = df.merge(df_review_counts[['drug_name', 'number_of_reviews']], 
              on='drug_name', how='left')

# =========================
# DATA PROCESSING
# =========================

numeric_cols = [
    "total_articles",
    "positive_studies_(%)",
    "negative_studies_(%)",
    "high_quality_evidence_(%)",
    "recent_publications_(2yr)",
    "avg_rating"
]

for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Calculate consensus gap
df["consensus_gap"] = (df["positive_studies_(%)"] - df["negative_studies_(%)"]).abs()

def consensus_label(gap):
    if pd.isna(gap):
        return "Insufficient data"
    if gap < 10:
        return "Strong consensus"
    elif gap < 25:
        return "Moderate disagreement"
    else:
        return "Polarized literature"

df["consensus_label"] = df["consensus_gap"].apply(consensus_label)

# Research freshness
df["recent_pub_share"] = df.apply(
    lambda r: r["recent_publications_(2yr)"] / r["total_articles"]
    if pd.notna(r["total_articles"]) and r["total_articles"] > 0
    else 0,
    axis=1
)

# Fill NaN values
df["overall_sentiment"] = df["overall_sentiment"].fillna("unknown")
df["all_side_effects"] = df["all_side_effects"].fillna("")
df["primary_safety_concern"] = df["primary_safety_concern"].fillna("Not specified")
df["literature_assessment"] = df.get("literature_assessment", pd.Series([""] * len(df))).fillna("")
df["most_positive_comment"] = df["most_positive_comment"].fillna("No positive comment available")
df["most_negative_comment"] = df["most_negative_comment"].fillna("No negative comment available")
df["number_of_reviews"] = pd.to_numeric(df["number_of_reviews"], errors="coerce").fillna(0).astype(int)

# =========================
# RISK SCORING
# =========================

def calculate_risk_score(row):
    score = 0
    
    if row["negative_studies_(%)"] > 40:
        score += 3
    elif row["negative_studies_(%)"] > 25:
        score += 2
    elif row["negative_studies_(%)"] > 15:
        score += 1
    
    if row["high_quality_evidence_(%)"] < 30:
        score += 2
    elif row["high_quality_evidence_(%)"] < 50:
        score += 1
    
    if pd.notna(row["avg_rating"]) and row["avg_rating"] < 2.5:
        score += 2
    elif pd.notna(row["avg_rating"]) and row["avg_rating"] < 3.5:
        score += 1
    
    # Sentiment-based risk
    if row["overall_sentiment"] == "negative":
        score += 2
    
    serious_keywords = ["heart attack", "death", "liver", "kidney", "stroke", "allergic"]
    if any(kw in str(row["primary_safety_concern"]).lower() for kw in serious_keywords):
        score += 2
    
    if score >= 7:
        return " High Risk", score
    elif score >= 4:
        return " Moderate Risk", score
    else:
        return " Low Risk", score

df["risk_label"], df["risk_score"] = zip(*df.apply(calculate_risk_score, axis=1))

# Extract side effects
def extract_side_effects(side_effects_str):
    if pd.isna(side_effects_str) or side_effects_str == "":
        return []
    return [e.strip().lower() for e in str(side_effects_str).split("|") if e.strip()]

df["side_effects_list"] = df["all_side_effects"].apply(extract_side_effects)

# =========================
# VISUALIZATION FUNCTIONS
# =========================

def create_sentiment_gauge(rating):
    if pd.isna(rating):
        rating = 0
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rating,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Patient Rating", 'font': {'size': 20, 'family': 'Archivo', 'weight': 700}},
        number={'font': {'size': 48, 'family': 'Space Mono', 'weight': 700}},
        gauge={
            'axis': {'range': [None, 5], 'tickwidth': 2, 'tickcolor': "#1a1a1a"},
            'bar': {'color': "#0066FF", 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#e0e0e0",
            'steps': [
                {'range': [0, 2], 'color': "#FFE5ED"},
                {'range': [2, 3.5], 'color': "#FFF5E5"},
                {'range': [3.5, 5], 'color': "#E5F9E5"}
            ],
            'threshold': {
                'line': {'color': "#FF3366", 'width': 4},
                'thickness': 0.85,
                'value': 4
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Archivo'}
    )
    return fig

def create_side_effects_chart(side_effects_list):
    if not side_effects_list or len(side_effects_list) == 0:
        return go.Figure().add_annotation(
            text="No side effects data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#999", family="Archivo")
        )
    
    effects_count = Counter(side_effects_list)
    top_effects = dict(effects_count.most_common(10))
    
    colors = ['#FF3366' if i % 2 == 0 else '#FF6B8A' for i in range(len(top_effects))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(top_effects.values()),
            y=list(top_effects.keys()),
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='#1a1a1a', width=2)
            ),
            text=list(top_effects.values()),
            textposition='outside',
            textfont=dict(size=14, family='Space Mono', weight=700)
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="Top Reported Side Effects",
            font=dict(size=20, family='Archivo', weight=700, color='#1a1a1a')
        ),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor='#f0f0f0',
            zeroline=False
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=13, family='Archivo', weight=600)
        ),
        height=400,
        margin=dict(l=20, r=80, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        font={'family': 'Archivo'}
    )
    
    return fig

def create_wordcloud_from_effects(side_effects_list):
    if not side_effects_list or len(side_effects_list) == 0:
        return None
    
    text = " ".join(side_effects_list)
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='RdPu',
        relative_scaling=0.5,
        min_font_size=12,
        font_path=None,
        contour_width=2,
        contour_color='#1a1a1a'
    ).generate(text)
    
    img = BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    encoded = base64.b64encode(img.read()).decode()
    
    return f"data:image/png;base64,{encoded}"

def create_comparison_chart(drug1_data, drug2_data, drug1_name, drug2_name):
    categories = [
        'Positive<br>Studies',
        'Evidence<br>Quality',
        'Patient<br>Rating (×20)',
        'Safety<br>(100-Neg%)'
    ]
    
    drug1_values = [
        drug1_data["positive_studies_(%)"],
        drug1_data["high_quality_evidence_(%)"],
        drug1_data["avg_rating"] * 20 if pd.notna(drug1_data["avg_rating"]) else 0,
        100 - drug1_data["negative_studies_(%)"]
    ]
    
    drug2_values = [
        drug2_data["positive_studies_(%)"],
        drug2_data["high_quality_evidence_(%)"],
        drug2_data["avg_rating"] * 20 if pd.notna(drug2_data["avg_rating"]) else 0,
        100 - drug2_data["negative_studies_(%)"]
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=drug1_values,
        theta=categories,
        fill='toself',
        name=drug1_name,
        line=dict(color='#0066FF', width=3),
        fillcolor='rgba(0, 102, 255, 0.2)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=drug2_values,
        theta=categories,
        fill='toself',
        name=drug2_name,
        line=dict(color='#FF3366', width=3),
        fillcolor='rgba(255, 51, 102, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=12, family='Space Mono', weight=700),
                gridcolor='#e0e0e0'
            ),
            angularaxis=dict(
                tickfont=dict(size=14, family='Archivo', weight=700)
            )
        ),
        showlegend=True,
        legend=dict(
            font=dict(size=16, family='Archivo', weight=700),
            orientation='h',
            yanchor='bottom',
            y=1.1,
            xanchor='center',
            x=0.5
        ),
        height=550,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Archivo'}
    )
    
    return fig

def generate_strategic_insights(drug_data):
    insights = []
    
    # Sentiment-based insight
    if drug_data['overall_sentiment'] == 'negative':
        insights.append(f" Overall sentiment is NEGATIVE - patient experience issues need immediate attention")
    elif drug_data['overall_sentiment'] == 'positive':
        insights.append(f" Overall sentiment is POSITIVE - leverage patient satisfaction in marketing")
    
    if pd.notna(drug_data['avg_rating']):
        if drug_data['avg_rating'] >= 4.0:
            insights.append(f" Strong patient satisfaction ({drug_data['avg_rating']:.1f}/5) - maintain current formulation")
        elif drug_data['avg_rating'] < 3.0:
            insights.append(f" Low patient satisfaction ({drug_data['avg_rating']:.1f}/5) - reformulation recommended")
    
    if drug_data['high_quality_evidence_(%)'] < 40:
        insights.append(f" Evidence quality is low ({drug_data['high_quality_evidence_(%)']:.0f}%) - invest in rigorous clinical trials")
    elif drug_data['high_quality_evidence_(%)'] > 70:
        insights.append(f" Strong evidence base ({drug_data['high_quality_evidence_(%)']:.0f}%) - leverage in professional marketing")
    
    if drug_data['negative_studies_(%)'] > 30:
        insights.append(f" Safety signals detected ({drug_data['negative_studies_(%)']:.0f}% negative studies) - proactive communication required")
    elif drug_data['negative_studies_(%)'] < 15:
        insights.append(f" Favorable safety profile ({drug_data['negative_studies_(%)']:.0f}% negative studies) - emphasize in positioning")
    
    if drug_data['recent_pub_share'] > 0.4:
        insights.append(f"📈 High research momentum ({drug_data['recent_pub_share']:.0%} recent publications) - emerging interest")
    elif drug_data['recent_pub_share'] < 0.2 and drug_data['total_articles'] > 10:
        insights.append(f"📉 Declining research activity ({drug_data['recent_pub_share']:.0%} recent publications) - waning clinical interest")
    
    return "\n\n".join([f"{i+1}. {insight}" for i, insight in enumerate(insights[:5])])

# =========================
# APP SETUP & STYLES
# =========================

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Archivo:wght@300;400;600;700;900&display=swap" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Archivo', -apple-system, sans-serif;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh;
            }
            
            .container {
                max-width: 1600px;
                margin: 0 auto;
                padding: 40px 20px;
            }
            
            .header {
                text-align: center;
                margin-bottom: 50px;
                animation: fadeInDown 0.8s ease;
            }
            
            .header h1 {
                font-size: 3.5em;
                font-weight: 900;
                background: linear-gradient(135deg, #0066FF, #00D4AA);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 10px;
                font-family: 'Archivo', sans-serif;
                letter-spacing: -2px;
            }
            
            .header p {
                font-size: 1.3em;
                color: #1a1a1a;
                font-weight: 600;
                font-family: 'Space Mono', monospace;
            }
            
            .tabs {
                display: flex;
                gap: 8px;
                margin-bottom: 40px;
                background: white;
                padding: 8px;
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
                animation: fadeIn 1s ease;
            }
            
            .tabs .tab {
                flex: 1;
                padding: 16px 24px;
                background: transparent;
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: 700;
                font-family: 'Archivo', sans-serif;
                color: #666;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .tabs .tab--selected {
                background: linear-gradient(135deg, #0066FF, #00D4AA);
                color: white;
                box-shadow: 0 4px 16px rgba(0, 102, 255, 0.3);
            }
            
            .dropdown {
                margin-bottom: 40px;
                animation: fadeIn 1.2s ease;
            }
            
            .Select-control {
                border: 3px solid #e0e0e0 !important;
                border-radius: 16px !important;
                padding: 8px !important;
                font-size: 18px !important;
                font-family: 'Archivo', sans-serif !important;
                font-weight: 700 !important;
                transition: all 0.3s ease !important;
            }
            
            .Select-control:hover {
                border-color: #0066FF !important;
                box-shadow: 0 4px 20px rgba(0, 102, 255, 0.15) !important;
            }
            
            .is-focused .Select-control {
                border-color: #0066FF !important;
                box-shadow: 0 4px 20px rgba(0, 102, 255, 0.2) !important;
            }
            
            .card {
                background: white;
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
                border: 3px solid transparent;
                transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                position: relative;
                overflow: hidden;
            }
            
            .card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #0066FF, #00D4AA);
                transform: scaleX(0);
                transform-origin: left;
                transition: transform 0.4s ease;
            }
            
            .card:hover::before {
                transform: scaleX(1);
            }
            
            .card:hover {
                transform: translateY(-4px);
                box-shadow: 0 16px 48px rgba(0, 0, 0, 0.12);
                border-color: rgba(0, 102, 255, 0.2);
            }
            
            .risk-high {
                border-color: #FF3366 !important;
                background: linear-gradient(135deg, #fff5f7, #ffffff);
            }
            
            .risk-moderate {
                border-color: #FFB020 !important;
                background: linear-gradient(135deg, #fffbf0, #ffffff);
            }
            
            .risk-low {
                border-color: #00D4AA !important;
                background: linear-gradient(135deg, #f0fff9, #ffffff);
            }
            
            .alert {
                padding: 24px;
                border-radius: 16px;
                margin-bottom: 30px;
                border-left: 6px solid;
                font-weight: 600;
                animation: slideInLeft 0.6s ease;
            }
            
            .alert-critical {
                background: linear-gradient(135deg, #FFE5ED, #FFF0F5);
                border-left-color: #FF3366;
                color: #8B1538;
            }
            
            .alert-warning {
                background: linear-gradient(135deg, #FFF5E5, #FFFBF0);
                border-left-color: #FFB020;
                color: #8B5A00;
            }
            
            .alert-info {
                background: linear-gradient(135deg, #E5F5FF, #F0F9FF);
                border-left-color: #0066FF;
                color: #003D99;
            }
            
            .comment-box {
                background: white;
                border-radius: 16px;
                padding: 24px;
                border: 3px solid;
                margin-bottom: 20px;
                position: relative;
                animation: fadeInUp 0.8s ease;
            }
            
            .comment-box::before {
                content: '';
                position: absolute;
                top: -3px;
                left: -3px;
                right: -3px;
                bottom: -3px;
                background: linear-gradient(135deg, currentColor, transparent);
                border-radius: 16px;
                z-index: -1;
                opacity: 0.1;
            }
            
            .comment-positive {
                border-color: #00D4AA;
            }
            
            .comment-negative {
                border-color: #FF3366;
            }
            
            .comment-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 16px;
            }
            
            .comment-label {
                font-size: 1.1em;
                font-weight: 900;
                font-family: 'Space Mono', monospace;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .comment-text {
                color: #1a1a1a;
                line-height: 1.8;
                font-size: 1.05em;
                font-style: italic;
                padding: 16px;
                background: rgba(0, 0, 0, 0.02);
                border-radius: 12px;
                border-left: 4px solid currentColor;
            }
            
            .grid-2 {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 24px;
                margin-bottom: 30px;
            }
            
            .grid-3 {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 24px;
                margin-bottom: 30px;
            }
            
            .grid-4 {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .metric-card {
                background: white;
                border-radius: 16px;
                padding: 24px;
                border: 3px solid #e0e0e0;
                text-align: center;
                transition: all 0.3s ease;
            }
            
            .metric-card:hover {
                transform: translateY(-4px);
                border-color: #0066FF;
                box-shadow: 0 8px 24px rgba(0, 102, 255, 0.15);
            }
            
            .metric-label {
                font-size: 0.85em;
                color: #666;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 8px;
                font-family: 'Space Mono', monospace;
            }
            
            .metric-value {
                font-size: 2.5em;
                font-weight: 900;
                color: #1a1a1a;
                font-family: 'Space Mono', monospace;
            }
            
            .metric-value-small {
                font-size: 1.4em;
            }
            
            .insights-box {
                background: linear-gradient(135deg, #F0F8FF, #E5F5FF);
                border: 3px solid #0066FF;
                border-radius: 20px;
                padding: 32px;
                animation: fadeIn 1.4s ease;
            }
            
            .insights-title {
                font-size: 1.8em;
                font-weight: 900;
                margin-bottom: 24px;
                color: #1a1a1a;
                font-family: 'Archivo', sans-serif;
            }
            
            .insight-item {
                background: white;
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 16px;
                border-left: 5px solid #0066FF;
                font-size: 1.05em;
                line-height: 1.7;
                font-weight: 600;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
                transition: all 0.3s ease;
            }
            
            .insight-item:hover {
                transform: translateX(8px);
                box-shadow: 0 6px 20px rgba(0, 102, 255, 0.15);
            }
            
            .section-title {
                font-size: 1.8em;
                font-weight: 900;
                margin-bottom: 24px;
                color: #1a1a1a;
                font-family: 'Archivo', sans-serif;
                display: flex;
                align-items: center;
                gap: 12px;
            }
            
            .empty-state {
                text-align: center;
                padding: 100px 20px;
                font-size: 1.4em;
                color: #666;
                font-weight: 700;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            @keyframes fadeInDown {
                from {
                    opacity: 0;
                    transform: translateY(-30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @keyframes slideInLeft {
                from {
                    opacity: 0;
                    transform: translateX(-30px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
            
            @media (max-width: 968px) {
                .grid-2, .grid-3, .grid-4 {
                    grid-template-columns: 1fr;
                }
                
                .header h1 {
                    font-size: 2.5em;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# =========================
# LAYOUT
# =========================

app.layout = html.Div(className="container", children=[
    
    html.Div(className="header", children=[
        html.H1(" Pharma Perception Intel"),
        html.P("NLP-Powered Market Intelligence from 4000+ Drugs"),
    ]),

    dcc.Tabs(id="mode-tabs", value='single', className="tabs", children=[
        dcc.Tab(label=' Drug Analysis', value='single'),
        dcc.Tab(label=' Comparison', value='compare'),
    ]),

    html.Div(id="mode-content")
])

# =========================
# CALLBACKS
# =========================

@app.callback(
    Output("mode-content", "children"),
    Input("mode-tabs", "value")
)
def render_mode(tab):
    if tab == 'single':
        return html.Div([
            dcc.Dropdown(
                id="drug-dropdown",
                options=[{"label": d, "value": d} for d in sorted(df["drug_name"].unique())],
                placeholder="Select a drug to analyze...",
                clearable=False,
                className="dropdown"
            ),
            html.Div(id="single-drug-output")
        ])
    else:
        return html.Div([
            html.Div(className="grid-2", children=[
                dcc.Dropdown(
                    id="drug1-dropdown",
                    options=[{"label": d, "value": d} for d in sorted(df["drug_name"].unique())],
                    placeholder="Select first drug...",
                    clearable=False
                ),
                dcc.Dropdown(
                    id="drug2-dropdown",
                    options=[{"label": d, "value": d} for d in sorted(df["drug_name"].unique())],
                    placeholder="Select second drug...",
                    clearable=False
                ),
            ]),
            html.Div(id="comparison-output")
        ])

@app.callback(
    Output("single-drug-output", "children"),
    Input("drug-dropdown", "value")
)
def update_single_drug(drug):
    if not drug:
        return html.Div(" Select a drug to view comprehensive analysis", className="empty-state")

    r = df[df["drug_name"] == drug].iloc[0]
    
    # Format values
    avg_rating = f"{r['avg_rating']:.2f}" if pd.notna(r["avg_rating"]) else "N/A"
    sentiment = str(r["overall_sentiment"]).title()
    
    side_effects_list = r["side_effects_list"]
    top_5_effects = ", ".join(side_effects_list[:5]) if side_effects_list else "Not reported"
    
    # Visualizations
    gauge_fig = create_sentiment_gauge(r["avg_rating"])
    side_effects_fig = create_side_effects_chart(side_effects_list)
    wordcloud_img = create_wordcloud_from_effects(side_effects_list)
    
    strategic_insights = generate_strategic_insights(r)
    
    # Determine risk class
    risk_class = "risk-high" if "High" in r["risk_label"] else \
                 "risk-moderate" if "Moderate" in r["risk_label"] else "risk-low"
    
    return html.Div(children=[
        
        # Drug Header
        html.Div(className=f"card {risk_class}", style={'marginBottom': '30px'}, children=[
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}, children=[
                html.Div(children=[
                    html.H2(drug, style={'fontSize': '2.5em', 'marginBottom': '12px', 'fontWeight': '900'}),
                    html.Div(r["risk_label"], style={
                        'fontSize': '1.2em',
                        'fontWeight': '900',
                        'fontFamily': 'Space Mono, monospace'
                    })
                ]),
                html.Div(children=[
                    html.Div("Overall Sentiment", style={'fontSize': '0.9em', 'color': '#666', 'marginBottom': '8px', 'fontWeight': '700', 'textTransform': 'uppercase'}),
                    html.Div(sentiment, style={
                        'fontSize': '2em',
                        'fontWeight': '900',
                        'color': '#00D4AA' if sentiment == 'Positive' else '#FF3366' if sentiment == 'Negative' else '#FFB020'
                    })
                ])
            ])
        ]),
        
        # Best & Worst Comments
        html.Div(className="section-title", children=["", "Patient Voice Highlights"]),
        html.Div(className="grid-2", children=[
            html.Div(className="comment-box comment-positive", children=[
                html.Div(className="comment-header", children=[
                    html.Div(" Most Positive", className="comment-label", style={'color': '#00D4AA'}),
                ]),
                html.Div(r["most_positive_comment"], className="comment-text", style={'color': '#00D4AA'})
            ]),
            
            html.Div(className="comment-box comment-negative", children=[
                html.Div(className="comment-header", children=[
                    html.Div(" Most Negative", className="comment-label", style={'color': '#FF3366'}),
                ]),
                html.Div(r["most_negative_comment"], className="comment-text", style={'color': '#FF3366'})
            ])
        ]),
        
        # Total Reviews Metric
        html.Div(style={'marginTop': '20px', 'marginBottom': '30px'}, children=[
            html.Div(className="metric-card", style={'textAlign': 'center', 'maxWidth': '400px', 'margin': '0 auto'}, children=[
                html.Div("📊 Total Patient Reviews", className="metric-label"),
                html.Div(f"{int(r['number_of_reviews'])}" if pd.notna(r['number_of_reviews']) else "0", 
                        className="metric-value", 
                        style={'fontSize': '3em', 'color': '#0066FF'})
            ])
        ]),
        
        # Key Metrics
        html.Div(className="section-title", children=["", "Clinical Literature Metrics"]),
        html.Div(className="grid-4", children=[
            html.Div(className="metric-card", children=[
                html.Div(" Total Studies", className="metric-label"),
                html.Div(f"{int(r['total_articles'])}", className="metric-value")
            ]),
            html.Div(className="metric-card", children=[
                html.Div(" Evidence Quality", className="metric-label"),
                html.Div(f"{r['high_quality_evidence_(%)']:.1f}%", className="metric-value")
            ]),
            html.Div(className="metric-card", children=[
                html.Div(" Positive Studies", className="metric-label"),
                html.Div(f"{r['positive_studies_(%)']:.0f}%", className="metric-value", style={'color': '#00D4AA'})
            ]),
            html.Div(className="metric-card", children=[
                html.Div(" Negative Studies", className="metric-label"),
                html.Div(f"{r['negative_studies_(%)']:.0f}%", className="metric-value", style={'color': '#FF3366'})
            ]),
        ]),
        
        # Visualizations
        html.Div(className="section-title", children=["", "Visual Analytics"]),
        html.Div(className="grid-2", children=[
            html.Div(className="card", children=[
                dcc.Graph(figure=gauge_fig, config={'displayModeBar': False})
            ]),
            html.Div(className="card", children=[
                html.Div(" Primary Safety Concern", style={'fontSize': '1.3em', 'fontWeight': '700', 'marginBottom': '16px'}),
                html.P(r["primary_safety_concern"], style={'fontSize': '1.1em', 'lineHeight': '1.6', 'marginBottom': '20px'}),
                html.Hr(style={'margin': '20px 0', 'border': 'none', 'borderTop': '2px solid #e0e0e0'}),
                html.Div("🩺 Top Reported Side Effects", style={'fontSize': '1.1em', 'fontWeight': '700', 'marginBottom': '12px'}),
                html.P(top_5_effects, style={'fontSize': '1.05em', 'color': '#666', 'lineHeight': '1.6'})
            ])
        ]),
        
        html.Div(className="grid-2", children=[
            html.Div(className="card", children=[
                dcc.Graph(figure=side_effects_fig, config={'displayModeBar': False})
            ]),
            html.Div(className="card", children=[
                html.Div(" Side Effects Word Cloud", style={'fontSize': '1.3em', 'fontWeight': '700', 'marginBottom': '16px'}),
                html.Img(src=wordcloud_img, style={'width': '100%', 'borderRadius': '12px'}) if wordcloud_img else 
                html.P("No side effects data available", style={'textAlign': 'center', 'color': '#999', 'padding': '60px 20px', 'fontSize': '1.1em'})
            ])
        ]),
        
        # Strategic Insights
        html.Div(className="insights-box", children=[
            html.Div(" AI-Powered Strategic Insights", className="insights-title"),
            html.Div([
                html.Div(insight, className="insight-item")
                for insight in strategic_insights.split('\n\n')
            ])
        ])
    ])

@app.callback(
    Output("comparison-output", "children"),
    Input("drug1-dropdown", "value"),
    Input("drug2-dropdown", "value")
)
def update_comparison(drug1, drug2):
    if not drug1 or not drug2:
        return html.Div(" Select two drugs to compare", className="empty-state")
    
    if drug1 == drug2:
        return html.Div(" Please select two different drugs", className="empty-state")
    
    d1 = df[df["drug_name"] == drug1].iloc[0]
    d2 = df[df["drug_name"] == drug2].iloc[0]
    
    comparison_fig = create_comparison_chart(d1, d2, drug1, drug2)
    
    return html.Div(children=[
        
        html.Div(className="card", style={'marginBottom': '30px'}, children=[
            html.H3(f"{drug1} vs {drug2}", style={'textAlign': 'center', 'fontSize': '2.2em', 'fontWeight': '900', 'marginBottom': '30px'}),
            dcc.Graph(figure=comparison_fig, config={'displayModeBar': False})
        ]),
        
        html.Div(className="grid-2", children=[
            
            # Drug 1
            html.Div(className="card", style={'borderColor': '#0066FF'}, children=[
                html.H4(drug1, style={'fontSize': '1.8em', 'fontWeight': '900', 'marginBottom': '24px', 'color': '#0066FF'}),
                
                html.Div(style={'marginBottom': '16px', 'paddingBottom': '16px', 'borderBottom': '2px solid #f0f0f0'}, children=[
                    html.Div("Patient Rating", style={'fontSize': '0.9em', 'color': '#666', 'marginBottom': '8px', 'fontWeight': '700'}),
                    html.Div(f"{d1['avg_rating']:.2f}/5" if pd.notna(d1['avg_rating']) else "N/A", 
                            style={'fontSize': '2em', 'fontWeight': '900', 'fontFamily': 'Space Mono, monospace'})
                ]),
                html.Div(style={'marginBottom': '16px', 'paddingBottom': '16px', 'borderBottom': '2px solid #f0f0f0'}, children=[
                    html.Div("Overall Sentiment", style={'fontSize': '0.9em', 'color': '#666', 'marginBottom': '8px', 'fontWeight': '700'}),
                    html.Div(d1['overall_sentiment'].title(), 
                            style={'fontSize': '1.5em', 'fontWeight': '900', 
                                  'color': '#00D4AA' if d1['overall_sentiment'] == 'positive' else '#FF3366' if d1['overall_sentiment'] == 'negative' else '#FFB020'})
                ]),
                html.Div(style={'marginBottom': '16px', 'paddingBottom': '16px', 'borderBottom': '2px solid #f0f0f0'}, children=[
                    html.Div("Risk Level", style={'fontSize': '0.9em', 'color': '#666', 'marginBottom': '8px', 'fontWeight': '700'}),
                    html.Div(d1['risk_label'], style={'fontSize': '1.2em', 'fontWeight': '900'})
                ]),
                html.Div(style={'marginBottom': '16px', 'paddingBottom': '16px', 'borderBottom': '2px solid #f0f0f0'}, children=[
                    html.Div("Evidence Quality", style={'fontSize': '0.9em', 'color': '#666', 'marginBottom': '8px', 'fontWeight': '700'}),
                    html.Div(f"{d1['high_quality_evidence_(%)']:.1f}%", style={'fontSize': '2em', 'fontWeight': '900', 'fontFamily': 'Space Mono, monospace'})
                ]),
            ]),
            
            # Drug 2
            html.Div(className="card", style={'borderColor': '#FF3366'}, children=[
                html.H4(drug2, style={'fontSize': '1.8em', 'fontWeight': '900', 'marginBottom': '24px', 'color': '#FF3366'}),
                
                html.Div(style={'marginBottom': '16px', 'paddingBottom': '16px', 'borderBottom': '2px solid #f0f0f0'}, children=[
                    html.Div("Patient Rating", style={'fontSize': '0.9em', 'color': '#666', 'marginBottom': '8px', 'fontWeight': '700'}),
                    html.Div(f"{d2['avg_rating']:.2f}/5" if pd.notna(d2['avg_rating']) else "N/A", 
                            style={'fontSize': '2em', 'fontWeight': '900', 'fontFamily': 'Space Mono, monospace'})
                ]),
                html.Div(style={'marginBottom': '16px', 'paddingBottom': '16px', 'borderBottom': '2px solid #f0f0f0'}, children=[
                    html.Div("Overall Sentiment", style={'fontSize': '0.9em', 'color': '#666', 'marginBottom': '8px', 'fontWeight': '700'}),
                    html.Div(d2['overall_sentiment'].title(), 
                            style={'fontSize': '1.5em', 'fontWeight': '900',
                                  'color': '#00D4AA' if d2['overall_sentiment'] == 'positive' else '#FF3366' if d2['overall_sentiment'] == 'negative' else '#FFB020'})
                ]),
                html.Div(style={'marginBottom': '16px', 'paddingBottom': '16px', 'borderBottom': '2px solid #f0f0f0'}, children=[
                    html.Div("Risk Level", style={'fontSize': '0.9em', 'color': '#666', 'marginBottom': '8px', 'fontWeight': '700'}),
                    html.Div(d2['risk_label'], style={'fontSize': '1.2em', 'fontWeight': '900'})
                ]),
                html.Div(style={'marginBottom': '16px', 'paddingBottom': '16px', 'borderBottom': '2px solid #f0f0f0'}, children=[
                    html.Div("Evidence Quality", style={'fontSize': '0.9em', 'color': '#666', 'marginBottom': '8px', 'fontWeight': '700'}),
                    html.Div(f"{d2['high_quality_evidence_(%)']:.1f}%", style={'fontSize': '2em', 'fontWeight': '900', 'fontFamily': 'Space Mono, monospace'})
                ]),
            ])
        ]),
        
        html.Div(className="card", style={'marginTop': '30px', 'background': 'linear-gradient(135deg, #FFFBF0, #FFF)'}, children=[
            html.Div(" Key Differences", style={'fontSize': '1.6em', 'fontWeight': '900', 'marginBottom': '20px'}),
            html.Ul([
                html.Li(f"Patient Satisfaction: {drug1} ({d1['avg_rating']:.2f}/5) vs {drug2} ({d2['avg_rating']:.2f}/5)", 
                       style={'padding': '12px 0', 'fontSize': '1.1em', 'fontWeight': '600'}),
                html.Li(f"Sentiment: {drug1} ({d1['overall_sentiment']}) vs {drug2} ({d2['overall_sentiment']})", 
                       style={'padding': '12px 0', 'fontSize': '1.1em', 'fontWeight': '600'}),
                html.Li(f"Evidence Quality: {drug1} ({d1['high_quality_evidence_(%)']:.1f}%) vs {drug2} ({d2['high_quality_evidence_(%)']:.1f}%)", 
                       style={'padding': '12px 0', 'fontSize': '1.1em', 'fontWeight': '600'}),
                html.Li(f"Safety Profile: {drug1} is {d1['risk_label']} vs {drug2} is {d2['risk_label']}", 
                       style={'padding': '12px 0', 'fontSize': '1.1em', 'fontWeight': '600'}),
            ], style={'listStyle': 'none', 'padding': '0'})
        ])
    ])

if __name__ == "__main__":
    app.run(port=8054, debug=True)
