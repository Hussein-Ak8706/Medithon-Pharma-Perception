#external implementation with css
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

df = pd.read_csv("Datasets/rahul_final_enriched.csv")
df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

# =========================
# CLEAN + DERIVED FEATURES
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

# Consensus gap
df["consensus_gap"] = (
    df["positive_studies_(%)"] - df["negative_studies_(%)"]
).abs()

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

# Text fields safety
df["overall_sentiment"] = df["overall_sentiment"].fillna("unknown")
df["all_side_effects"] = df["all_side_effects"].fillna("")
df["primary_safety_concern"] = df["primary_safety_concern"].fillna("Not specified")

# =========================
# PERCEPTION GAP ANALYSIS
# =========================

def calculate_perception_gap(row):
    """Detect mismatch between patient rating and literature sentiment"""
    rating = row["avg_rating"]
    pos_studies = row["positive_studies_(%)"]
    
    if pd.isna(rating) or pd.isna(pos_studies):
        return "Insufficient data"
    
    if rating > 4.0 and pos_studies < 40:
        return "⚠️ Rating-Literature Gap: Patients rate highly despite mixed evidence"
    elif rating < 3.0 and pos_studies > 60:
        return "⚠️ Rating-Literature Gap: Poor ratings despite positive literature"
    elif rating > 4.0 and pos_studies > 60:
        return "✅ Aligned: Strong patient satisfaction backed by evidence"
    else:
        return "Neutral alignment"

df["perception_gap"] = df.apply(calculate_perception_gap, axis=1)

# =========================
# RISK SCORING
# =========================

def calculate_risk_score(row):
    """Proprietary risk score based on multiple factors"""
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
    
    serious_keywords = ["heart attack", "death", "liver", "kidney", "stroke"]
    if any(kw in str(row["primary_safety_concern"]).lower() for kw in serious_keywords):
        score += 2
    
    if score >= 7:
        return "🔴 High Risk", score
    elif score >= 4:
        return "🟡 Moderate Risk", score
    else:
        return "🟢 Low Risk", score

df["risk_label"], df["risk_score"] = zip(*df.apply(calculate_risk_score, axis=1))

# =========================
# SIDE EFFECT ANALYSIS
# =========================

def extract_side_effects(side_effects_str):
    """Parse side effects from pipe-separated string"""
    if pd.isna(side_effects_str) or side_effects_str == "":
        return []
    return [e.strip().lower() for e in str(side_effects_str).split("|") if e.strip()]

df["side_effects_list"] = df["all_side_effects"].apply(extract_side_effects)

# =========================
# VISUALIZATION FUNCTIONS
# =========================

def create_sentiment_gauge(rating):
    """Create a gauge chart for patient rating"""
    if pd.isna(rating):
        rating = 0
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rating,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Patient Rating"},
        gauge={
            'axis': {'range': [None, 5]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 2], 'color': "#ffcccc"},
                {'range': [2, 3.5], 'color': "#ffffcc"},
                {'range': [3.5, 5], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 4
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_side_effects_chart(side_effects_list):
    """Create bar chart of side effects"""
    if not side_effects_list or len(side_effects_list) == 0:
        return go.Figure().add_annotation(
            text="No side effects data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    effects_count = Counter(side_effects_list)
    top_effects = dict(effects_count.most_common(10))
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(top_effects.values()),
            y=list(top_effects.keys()),
            orientation='h',
            marker=dict(color='#FF6B6B')
        )
    ])
    
    fig.update_layout(
        title="Top Reported Side Effects",
        xaxis_title="Mentions",
        yaxis_title="",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_wordcloud(side_effects_list):
    """Generate word cloud from side effects"""
    if not side_effects_list or len(side_effects_list) == 0:
        return None
    
    text = " ".join(side_effects_list)
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='Reds',
        relative_scaling=0.5,
        min_font_size=10
    ).generate(text)
    
    img = BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    encoded = base64.b64encode(img.read()).decode()
    
    return f"data:image/png;base64,{encoded}"

def create_comparison_chart(drug1_data, drug2_data, drug1_name, drug2_name):
    """Create radar chart comparing two drugs"""
    categories = [
        'Positive Studies',
        'Evidence Quality',
        'Patient Rating (×20)',
        'Research Freshness (×100)',
        'Safety (100-Neg%)'
    ]
    
    drug1_values = [
        drug1_data["positive_studies_(%)"],
        drug1_data["high_quality_evidence_(%)"],
        drug1_data["avg_rating"] * 20 if pd.notna(drug1_data["avg_rating"]) else 0,
        drug1_data["recent_pub_share"] * 100,
        100 - drug1_data["negative_studies_(%)"]
    ]
    
    drug2_values = [
        drug2_data["positive_studies_(%)"],
        drug2_data["high_quality_evidence_(%)"],
        drug2_data["avg_rating"] * 20 if pd.notna(drug2_data["avg_rating"]) else 0,
        drug2_data["recent_pub_share"] * 100,
        100 - drug2_data["negative_studies_(%)"]
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=drug1_values,
        theta=categories,
        fill='toself',
        name=drug1_name
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=drug2_values,
        theta=categories,
        fill='toself',
        name=drug2_name
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=500
    )
    
    return fig

# =========================
# STRATEGIC INSIGHTS
# =========================

def generate_strategic_insights(drug_data):
    """Generate insights using rule-based logic"""
    insights = []
    
    if pd.notna(drug_data['avg_rating']):
        if drug_data['avg_rating'] >= 4.0:
            insights.append(f"✅ Strong patient satisfaction ({drug_data['avg_rating']:.1f}/5) - maintain current formulation and marketing approach")
        elif drug_data['avg_rating'] < 3.0:
            insights.append(f"⚠️ Low patient satisfaction ({drug_data['avg_rating']:.1f}/5) - investigate common complaints and consider reformulation")
        else:
            insights.append(f"📊 Moderate patient satisfaction ({drug_data['avg_rating']:.1f}/5) - opportunity to improve user experience")
    
    if drug_data['high_quality_evidence_(%)'] < 40:
        insights.append(f"🔬 Evidence quality is low ({drug_data['high_quality_evidence_(%)']:.0f}%) - invest in rigorous clinical trials to strengthen market position")
    elif drug_data['high_quality_evidence_(%)'] > 70:
        insights.append(f"💎 Strong evidence base ({drug_data['high_quality_evidence_(%)']:.0f}%) - leverage in marketing to healthcare professionals")
    
    if drug_data['negative_studies_(%)'] > 30:
        top_concern = drug_data['primary_safety_concern']
        insights.append(f"⚠️ Safety signals detected ({drug_data['negative_studies_(%)']:.0f}% negative studies) - address '{top_concern}' proactively in communications")
    elif drug_data['negative_studies_(%)'] < 15:
        insights.append(f"✅ Favorable safety profile ({drug_data['negative_studies_(%)']:.0f}% negative studies) - emphasize in competitive positioning")
    
    if drug_data['recent_pub_share'] > 0.4:
        insights.append(f"📈 High research momentum ({drug_data['recent_pub_share']:.0%} recent publications) - emerging interest in the market")
    elif drug_data['recent_pub_share'] < 0.2:
        insights.append(f"📉 Declining research activity ({drug_data['recent_pub_share']:.0%} recent publications) - may indicate waning clinical interest")
    
    if "⚠️" in drug_data['perception_gap']:
        insights.append(f"🎯 {drug_data['perception_gap'].replace('⚠️ ', '')} - investigate root causes")
    
    return "\n\n".join([f"{i+1}. {insight}" for i, insight in enumerate(insights[:5])])

# =========================
# APP SETUP
# =========================

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# =========================
# LAYOUT
# =========================

app.layout = html.Div(className="container", children=[
    
    html.Div(className="header", children=[
        html.H1("🧬 Pharma Market Perception Intelligence"),
        html.P("AI-powered insights from 4000+ drugs across clinical literature and patient reviews"),
    ]),

    dcc.Tabs(id="mode-tabs", value='single', className="tabs", children=[
        dcc.Tab(label='📊 Single Drug Analysis', value='single'),
        dcc.Tab(label='⚖️ Comparative Analysis', value='compare'),
    ]),

    html.Div(id="mode-content")
])

# =========================
# MODE SWITCHING
# =========================

@app.callback(
    Output("mode-content", "children"),
    Input("mode-tabs", "value")
)
def render_mode(tab):
    if tab == 'single':
        return html.Div(className="single-mode", children=[
            dcc.Dropdown(
                id="drug-dropdown",
                options=[{"label": d, "value": d} for d in sorted(df["drug_name"].unique())],
                placeholder="Select a drug...",
                clearable=False,
                className="dropdown"
            ),
            html.Div(id="single-drug-output")
        ])
    else:
        return html.Div(className="compare-mode", children=[
            html.Div(className="compare-dropdowns", children=[
                dcc.Dropdown(
                    id="drug1-dropdown",
                    options=[{"label": d, "value": d} for d in sorted(df["drug_name"].unique())],
                    placeholder="Select first drug...",
                    clearable=False,
                    className="dropdown"
                ),
                dcc.Dropdown(
                    id="drug2-dropdown",
                    options=[{"label": d, "value": d} for d in sorted(df["drug_name"].unique())],
                    placeholder="Select second drug...",
                    clearable=False,
                    className="dropdown"
                ),
            ]),
            html.Div(id="comparison-output")
        ])

# =========================
# SINGLE DRUG ANALYSIS
# =========================

@app.callback(
    Output("single-drug-output", "children"),
    Input("drug-dropdown", "value")
)
def update_single_drug(drug):
    if not drug:
        return html.Div(
            "👆 Select a drug to view comprehensive analysis",
            className="empty-state"
        )

    r = df[df["drug_name"] == drug].iloc[0]
    
    # Format values
    avg_rating = f"{r['avg_rating']:.2f}" if pd.notna(r["avg_rating"]) else "N/A"
    sentiment = str(r["overall_sentiment"]).title()
    
    side_effects_list = r["side_effects_list"]
    top_5_effects = ", ".join(side_effects_list[:5]) if side_effects_list else "Not prominently reported"
    
    # Generate visualizations
    gauge_fig = create_sentiment_gauge(r["avg_rating"])
    side_effects_fig = create_side_effects_chart(side_effects_list)
    wordcloud_img = create_wordcloud(side_effects_list)
    
    # Strategic insights
    strategic_insights = generate_strategic_insights(r)
    
    # Determine risk class
    risk_class = "alert-highrisk" if "High" in r["risk_label"] else \
                 "alert-moderate" if "Moderate" in r["risk_label"] else "alert-low"
    
    # Determine perception class
    perception_class = "alert-warning" if "⚠️" in r["perception_gap"] else "alert-info"
    
    return html.Div(children=[
        
        # Header Card
        html.Div(className=f"card {risk_class}", children=[
            html.H2(drug),
            html.Div(f"{r['risk_label']} - Risk Score: {r['risk_score']}/10")
        ]),
        
        # Perception Gap Alert
        html.Div(className=f"card {perception_class}", children=[
            html.H4("🔍 Perception Analysis"),
            html.P(r["perception_gap"])
        ]) if "⚠️" in r["perception_gap"] or "✅" in r["perception_gap"] else None,
        
        # Key Metrics Grid
        html.Div(className="grid grid-auto-fit", children=[
            html.Div(className="card", children=[
                html.Div("📚 Total Studies", className="label"),
                html.Div(f"{int(r['total_articles'])}", className="value")
            ]),
            
            html.Div(className="card", children=[
                html.Div("✅ Evidence Quality", className="label"),
                html.Div(f"{r['high_quality_evidence_(%)']:.1f}%", className="value")
            ]),
            
            html.Div(className="card", children=[
                html.Div("📊 Study Sentiment", className="label"),
                html.Div(
                    f"{r['positive_studies_(%)']:.0f}% pos / {r['negative_studies_(%)']:.0f}% neg",
                    className="value value-small"
                )
            ]),
            
            html.Div(className="card", children=[
                html.Div("🤝 Consensus", className="label"),
                html.Div(r['consensus_label'], className="value value-small")
            ]),
            
            html.Div(className="card", children=[
                html.Div("🔬 Research Freshness", className="label"),
                html.Div(f"{r['recent_pub_share']:.1%}", className="value")
            ]),
            
            html.Div(className="card", children=[
                html.Div("💬 Patient Sentiment", className="label"),
                html.Div(sentiment, className="value")
            ]),
        ]),
        
        # Gauge + Safety Info
        html.Div(className="grid grid-2-3", children=[
            html.Div(className="card", children=[
                dcc.Graph(figure=gauge_fig, config={'displayModeBar': False})
            ]),
            
            html.Div(className="card", children=[
                html.H4("⚠️ Primary Safety Concern"),
                html.P(r["primary_safety_concern"], className="safety-text"),
                html.Hr(),
                html.Div("🩺 Top Reported Side Effects", className="label"),
                html.P(top_5_effects, className="side-effects-text")
            ])
        ]),
        
        # Side Effects Visualizations
        html.Div(className="grid grid-2", children=[
            html.Div(className="card", children=[
                dcc.Graph(figure=side_effects_fig, config={'displayModeBar': False})
            ]),
            
            html.Div(className="card", children=[
                html.H4("☁️ Side Effects Word Cloud"),
                html.Img(src=wordcloud_img, className="wordcloud-img") if wordcloud_img else 
                html.P("No side effects data available", className="no-data")
            ])
        ]),
        
        # Strategic Insights
        html.Div(className="card insights-card", children=[
            html.H4("🎯 Strategic Insights"),
            html.Pre(strategic_insights, className="insights-text")
        ])
    ])

# =========================
# COMPARATIVE ANALYSIS
# =========================

@app.callback(
    Output("comparison-output", "children"),
    Input("drug1-dropdown", "value"),
    Input("drug2-dropdown", "value")
)
def update_comparison(drug1, drug2):
    if not drug1 or not drug2:
        return html.Div(
            "👆 Select two drugs to compare",
            className="empty-state"
        )
    
    if drug1 == drug2:
        return html.Div(
            "⚠️ Please select two different drugs",
            className="error-state"
        )
    
    d1 = df[df["drug_name"] == drug1].iloc[0]
    d2 = df[df["drug_name"] == drug2].iloc[0]
    
    # Create comparison radar chart
    comparison_fig = create_comparison_chart(d1, d2, drug1, drug2)
    
    return html.Div(children=[
        
        # Radar Chart
        html.Div(className="card", children=[
            html.H3(f"{drug1} vs {drug2}", className="comparison-title"),
            dcc.Graph(figure=comparison_fig, config={'displayModeBar': False})
        ]),
        
        # Side-by-side comparison
        html.Div(className="grid grid-2", children=[
            
            # Drug 1
            html.Div(className="card comparison-card", children=[
                html.H4(drug1, className="drug-title drug1-title"),
                
                html.Div(className="metric-row", children=[
                    html.Div("Patient Rating", className="label"),
                    html.Div(f"{d1['avg_rating']:.2f}/5" if pd.notna(d1['avg_rating']) else "N/A", 
                            className="value")
                ]),
                
                html.Div(className="metric-row", children=[
                    html.Div("Risk Level", className="label"),
                    html.Div(d1['risk_label'], className="value value-small")
                ]),
                
                html.Div(className="metric-row", children=[
                    html.Div("Evidence Quality", className="label"),
                    html.Div(f"{d1['high_quality_evidence_(%)']:.1f}%", className="value")
                ]),
                
                html.Div(className="metric-row", children=[
                    html.Div("Study Sentiment", className="label"),
                    html.Div(f"{d1['positive_studies_(%)']:.0f}% positive", className="value")
                ]),
                
                html.Hr(),
                html.Div("Top Side Effects", className="label section-label"),
                html.Ul([html.Li(e) for e in d1['side_effects_list'][:5]], 
                       className="side-effects-list")
            ]),
            
            # Drug 2
            html.Div(className="card comparison-card", children=[
                html.H4(drug2, className="drug-title drug2-title"),
                
                html.Div(className="metric-row", children=[
                    html.Div("Patient Rating", className="label"),
                    html.Div(f"{d2['avg_rating']:.2f}/5" if pd.notna(d2['avg_rating']) else "N/A", 
                            className="value")
                ]),
                
                html.Div(className="metric-row", children=[
                    html.Div("Risk Level", className="label"),
                    html.Div(d2['risk_label'], className="value value-small")
                ]),
                
                html.Div(className="metric-row", children=[
                    html.Div("Evidence Quality", className="label"),
                    html.Div(f"{d2['high_quality_evidence_(%)']:.1f}%", className="value")
                ]),
                
                html.Div(className="metric-row", children=[
                    html.Div("Study Sentiment", className="label"),
                    html.Div(f"{d2['positive_studies_(%)']:.0f}% positive", className="value")
                ]),
                
                html.Hr(),
                html.Div("Top Side Effects", className="label section-label"),
                html.Ul([html.Li(e) for e in d2['side_effects_list'][:5]], 
                       className="side-effects-list")
            ])
        ]),
        
        # Key Differences
        html.Div(className="card differences-card", children=[
            html.H4("📌 Key Differences"),
            html.Ul([
                html.Li(f"Evidence Quality: {drug1} ({d1['high_quality_evidence_(%)']:.1f}%) vs {drug2} ({d2['high_quality_evidence_(%)']:.1f}%)"),
                html.Li(f"Patient Satisfaction: {drug1} ({d1['avg_rating']:.2f}/5) vs {drug2} ({d2['avg_rating']:.2f}/5)"),
                html.Li(f"Safety Profile: {drug1} is {d1['risk_label']} vs {drug2} is {d2['risk_label']}"),
                html.Li(f"Consensus: {drug1} has {d1['consensus_label'].lower()} vs {drug2} has {d2['consensus_label'].lower()}")
            ], className="differences-list")
        ])
    ])

# =========================
# RUN
# =========================


if __name__ == "__main__":
    app.run(port=8055, debug=True)
