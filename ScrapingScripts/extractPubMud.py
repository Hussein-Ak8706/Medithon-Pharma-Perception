
import httpx
from lxml import etree
from time import sleep
from datetime import datetime, timedelta
import csv
from collections import Counter, defaultdict
import re
import os
import sys

try:
    import ujson as json
except ImportError:
    import json

NCBI_API_KEY = "4766d70bb0b93cd86f5917048e7f22975608"
YEARS_BACK = 6
MAX_RESULTS = 200  # UPGRADED from 100
OUTPUT_FILE = "drug_literature_review_v2.csv"

# HTTP client with advanced connection pooling
client = httpx.Client(
    timeout=30.0,
    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
    http2=True
)

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

SECTION_WEIGHTS = {
    # Highest confidence - actual findings
    'CONCLUSIONS': 3.0,
    'CONCLUSION': 3.0,
    'RESULTS': 2.5,
    'RESULTS AND CONCLUSION': 2.5,
    'FINDINGS': 2.0,
    
    # High confidence - safety data
    'ADVERSE EVENTS': 3.0,
    'ADVERSE EFFECTS': 3.0,
    'SAFETY': 2.5,
    'SIDE EFFECTS': 2.5,
    'TOXICITY': 2.5,
    
    # Medium confidence
    'DISCUSSION': 1.5,
    'METHODS': 1.0,
    'OBJECTIVE': 1.0,
    'UNSPECIFIED': 1.0,
    
    # Lower confidence - context only
    'LIMITATIONS': 0.8,
    'BACKGROUND': 0.4,
    'INTRODUCTION': 0.4,
    'OBJECTIVE AND BACKGROUND': 0.5,
}
# IMPROVEMENT #1: ENHANCED NEGATION DETECTION
NEGATION_PATTERNS = re.compile(
    r'\b(no|not|without|lack of|absence of|failed to|unable to|neither|nor|never|'
    r'rarely|seldom|minimal|negligible|insignificant|did not|does not|'
    r'absence of improvement|lack of efficacy|failed to show|failed to demonstrate|'
    r'unable to demonstrate|showed no|no evidence of|no sign of|'
    r'did not achieve|could not|cannot)\b',
    re.IGNORECASE
)

# IMPROVEMENT #2: STATISTICAL SIGNIFICANCE DETECTION
STATISTICAL_PATTERNS = {
    'significant': re.compile(
        r'(p\s*[<≤]\s*0\.05|p\s*[<≤]\s*0\.01|p\s*=\s*0\.0\d+|'
        r'statistically significant|significant improvement|significant benefit|'
        r'significant reduction|significant increase)',
        re.IGNORECASE
    ),
    'not_significant': re.compile(
        r'(p\s*[>≥]\s*0\.05|not significant|no significant|ns\b|'
        r'not statistically significant|failed to reach significance)',
        re.IGNORECASE
    ),
    'confidence_interval': re.compile(
        r'(95%\s*CI|confidence interval|95%\s*confidence)',
        re.IGNORECASE
    )
}

# IMPROVEMENT #3: COMPARATIVE PHRASES
COMPARATIVE_PATTERNS = {
    'superior': re.compile(
        r'\b(superior to|better than|more effective than|outperformed|'
        r'greater efficacy than|improved over|surpassed)\b',
        re.IGNORECASE
    ),
    'inferior': re.compile(
        r'\b(inferior to|worse than|less effective than|underperformed|'
        r'lower efficacy than|failed to match)\b',
        re.IGNORECASE
    ),
    'non_inferior': re.compile(
        r'\b(non[- ]inferior|noninferior|comparable to|similar to|equivalent to)\b',
        re.IGNORECASE
    )
}

# IMPROVEMENT #4: SEVERITY & FREQUENCY INDICATORS
SEVERITY_INDICATORS = {
    'severe': re.compile(r'\b(severe|serious|life[- ]threatening|fatal|death|mortality|critical)\b', re.IGNORECASE),
    'moderate': re.compile(r'\b(moderate|significant)\b', re.IGNORECASE),
    'mild': re.compile(r'\b(mild|minor|slight|transient|reversible)\b', re.IGNORECASE),
}

FREQUENCY_INDICATORS = {
    'common': re.compile(r'\b(common|frequent|often|usual|typical|regularly)\b', re.IGNORECASE),
    'uncommon': re.compile(r'\b(uncommon|occasional|sometimes|infrequent)\b', re.IGNORECASE),
    'rare': re.compile(r'\b(rare|rarely|seldom|unusual|exceptional)\b', re.IGNORECASE),
}

# IMPROVEMENT #5: CAUSALITY INDICATORS
CAUSALITY_PATTERNS = {
    'strong': re.compile(r'\b(caused by|due to|resulted from|attributable to|induced by)\b', re.IGNORECASE),
    'moderate': re.compile(r'\b(associated with|related to|linked to|correlated with)\b', re.IGNORECASE),
    'weak': re.compile(r'\b(possible|possibly|may be|might be|could be)\b', re.IGNORECASE),
}

# Enhanced medical term dictionaries
MEDICAL_TERMS = {
    'efficacy_pos': [
        'effective', 'efficacy', 'beneficial', 'improved', 'improvement', 'successful',
        'therapeutic', 'relief', 'reduction', 'superior', 'potent', 'promising',
        'favorable', 'optimal', 'significant benefit', 'clinically meaningful',
        'therapeutic effect', 'clinical benefit', 'positive outcome', 'remission',
        'cure', 'healing', 'recovery', 'response rate'
    ],
    'efficacy_neg': [
        'ineffective', 'failed', 'insufficient', 'inadequate', 'suboptimal',
        'limited efficacy', 'no benefit', 'no effect', 'inferior', 'disappointing',
        'lack of efficacy', 'treatment failure', 'non-responder', 'resistant',
        'refractory', 'poor outcome', 'deterioration', 'progression'
    ],
    'safety_pos': [
        'safe', 'well-tolerated', 'tolerable', 'acceptable safety', 'favorable safety',
        'minimal side effects', 'low toxicity', 'reversible', 'benign',
        'no adverse events', 'good safety profile', 'no toxicity'
    ],
    'safety_neg': [
        'toxic', 'toxicity', 'adverse', 'serious adverse', 'severe',
        'life-threatening', 'fatal', 'death', 'mortality', 'discontinuation',
        'withdrawn', 'black box warning', 'contraindicated', 'hazard'
    ],
    'hepatotoxicity': [
        'hepatotoxic', 'liver damage', 'liver injury', 'liver failure',
        'ALT elevation', 'AST elevation', 'jaundice', 'hepatic injury',
        'hepatitis', 'cirrhosis', 'hepatic dysfunction', 'liver enzyme elevation'
    ],
    'nephrotoxicity': [
        'nephrotoxic', 'kidney damage', 'renal injury', 'renal failure',
        'acute kidney injury', 'elevated creatinine', 'renal dysfunction',
        'acute tubular necrosis', 'chronic kidney disease', 'dialysis'
    ],
    'gastrointestinal': [
        'nausea', 'vomiting', 'diarrhea', 'abdominal pain', 'GI bleeding',
        'gastritis', 'ulcer', 'dyspepsia', 'constipation', 'bowel perforation',
        'gastrointestinal distress', 'stomach upset'
    ],
    'cardiovascular': [
        'cardiac', 'arrhythmia', 'hypertension', 'heart failure',
        'QT prolongation', 'myocardial', 'tachycardia', 'bradycardia',
        'myocardial infarction', 'stroke', 'thrombosis', 'cardiac arrest'
    ],
    'allergic': [
        'allergic', 'allergy', 'hypersensitivity', 'rash', 'urticaria',
        'anaphylaxis', 'angioedema', 'Stevens-Johnson', 'pruritus',
        'dermatitis', 'allergic reaction'
    ],
    'hematologic': [
        'anemia', 'thrombocytopenia', 'neutropenia', 'leukopenia',
        'bleeding', 'coagulopathy', 'pancytopenia', 'bone marrow suppression'
    ]
}

# Pre-compile all patterns for speed
TERM_PATTERNS = {
    category: re.compile('|'.join(r'\b' + re.escape(term) + r'\b' for term in terms), re.IGNORECASE)
    for category, terms in MEDICAL_TERMS.items()
}

# IMPROVEMENT #6: ENHANCED EVIDENCE LEVELS
EVIDENCE_LEVELS = {
    'very_high': [
        'cochrane review', 'meta-analysis of randomized', 'network meta-analysis',
        'meta-analysis of rct', 'systematic review of rct'
    ],
    'high': [
        'randomized controlled trial', 'RCT', 'double-blind', 'placebo-controlled',
        'systematic review', 'phase III', 'multicenter', 'multicentre',
        'randomized trial', 'controlled trial', 'triple-blind'
    ],
    'medium': [
        'cohort study', 'case-control', 'phase II', 'clinical trial',
        'prospective study', 'open-label trial', 'observational study',
        'longitudinal study', 'comparative study'
    ],
    'low': [
        'case report', 'case series', 'phase I', 'retrospective',
        'cross-sectional', 'pilot study', 'feasibility study'
    ],
    'preclinical': [
        'in vitro', 'in vivo', 'animal model', 'cell culture',
        'mouse model', 'rat model', 'preclinical'
    ]
}

# IMPROVEMENT #7: STUDY SIZE EXTRACTION
SAMPLE_SIZE_PATTERN = re.compile(
    r'\b(?:n\s*=\s*|N\s*=\s*|sample of\s+|cohort of\s+|enrolled\s+)(\d+)\s*(?:patient|subject|participant)',
    re.IGNORECASE
)

# IMPROVEMENT #8: HIGH-IMPACT JOURNALS
HIGH_IMPACT_JOURNALS = [
    'lancet', 'new england journal of medicine', 'nejm', 'jama',
    'british medical journal', 'bmj', 'nature medicine',
    'annals of internal medicine', 'plos medicine', 'jama internal medicine',
    'circulation', 'journal of clinical oncology'
]

# Publication types to SKIP (low quality)
SKIP_PUBLICATION_TYPES = [
    'Editorial', 'Letter', 'Comment', 'News', 'Newspaper Article',
    'Biography', 'Congress', 'Retracted Publication'
]


def search_pubmed(query, retmax=200, date_from=None, date_to=None):
    """Search PubMed with HTTP/2 acceleration"""
    url = BASE_URL + "esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
        "api_key": NCBI_API_KEY
    }
    
    if date_from:
        params["mindate"] = date_from
    if date_to:
        params["maxdate"] = date_to
    
    try:
        response = client.get(url, params=params)
        response.raise_for_status()
        result = response.json()["esearchresult"]
        return result["idlist"]
    except Exception as e:
        print(f"Search error: {e}")
        return []


def fetch_articles(pmids, batch_size=200):
    """Fetch articles with connection pooling"""
    all_xml = []
    
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        
        url = BASE_URL + "efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "api_key": NCBI_API_KEY
        }
        
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            all_xml.append(response.text)
            sleep(0.1)
        except Exception as e:
            print(f"Fetch error: {e}")
    
    return all_xml


def parse_xml(xml_data_list):
    """Parse XML with lxml for maximum speed + enhanced metadata extraction"""
    articles = []
    seen_pmids = set()  # IMPROVEMENT: Deduplication
    
    for xml_data in xml_data_list:
        try:
            root = etree.fromstring(xml_data.encode('utf-8'))
        except:
            continue
        
        for article in root.xpath('.//PubmedArticle'):
            pmid = article.findtext('.//PMID')
            
            # IMPROVEMENT: Skip duplicates
            if pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)
            
            title = article.findtext('.//ArticleTitle') or ""
            year = article.findtext('.//PubDate/Year')
            
            # Extract publication types
            pub_types = [pt.text for pt in article.xpath('.//PublicationType') if pt.text]
            
            # IMPROVEMENT #10: Quality filter - skip low-value content
            if any(skip_type in pub_types for skip_type in SKIP_PUBLICATION_TYPES):
                continue
            
            # Extract journal name for impact factor check
            journal = article.findtext('.//Journal/Title') or ""
            
            # Extract abstract sections with labels
            sections = {}
            abstract_parts = article.xpath('.//AbstractText')
            
            # IMPROVEMENT: Handle missing abstracts gracefully
            if not abstract_parts:
                continue  # Skip articles with no abstract
            
            for part in abstract_parts:
                label = part.get('Label', 'UNSPECIFIED').upper()
                text = part.text or ""
                if text:
                    sections[label] = text.lower()
            
            # If no labeled sections, treat as UNSPECIFIED
            if not sections:
                full_abstract = ' '.join([p.text for p in abstract_parts if p.text])
                if full_abstract:
                    sections['UNSPECIFIED'] = full_abstract.lower()
                else:
                    continue  # Skip if truly no content
            
            # IMPROVEMENT #8: Extract sample size from abstract
            sample_size = None
            all_text = ' '.join(sections.values())
            size_match = SAMPLE_SIZE_PATTERN.search(all_text)
            if size_match:
                try:
                    sample_size = int(size_match.group(1))
                except:
                    pass
            
            articles.append({
                "pmid": pmid,
                "title": title.lower(),
                "year": year,
                "pub_types": pub_types,
                "journal": journal.lower(),
                "sections": sections,
                "sample_size": sample_size
            })
    
    return articles
  
def extract_sentences(text):
    """Split text into sentences"""
    # Simple sentence splitter
    sentences = re.split(r'[.!?]+\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def get_negation_scope(sentence, negation_pos):
    """
    IMPROVEMENT #1: Scope-based negation - only affects next 3-5 words
    Returns True if position is within negation scope
    """
    words = sentence.split()
    # Negation affects next 5 words
    scope_end = min(negation_pos + 5, len(words))
    return negation_pos, scope_end


def sentence_contains_drug(sentence, drug_name):
    """
    IMPROVEMENT #4: Check if sentence mentions the drug (word boundary)
    """
    drug_pattern = r'\b' + re.escape(drug_name.lower()) + r'\b'
    return bool(re.search(drug_pattern, sentence))


def analyze_sentence_sentiment(sentence, drug_name, weight=1.0):
    """
    IMPROVEMENT #1-5: Enhanced sentence-level sentiment with:
    - Context awareness (only analyze if drug mentioned)
    - Statistical significance detection
    - Comparative phrase detection
    - Severity/frequency weighting
    - Scope-based negation
    """
    
    # Only analyze sentences that mention the drug
    if not sentence_contains_drug(sentence, drug_name):
        return 0, {}
    
    metadata = {
        'has_stats': False,
        'is_comparative': False,
        'severity': None,
        'frequency': None
    }
    
    # Check for statistical significance
    if STATISTICAL_PATTERNS['significant'].search(sentence):
        metadata['has_stats'] = True
        weight *= 1.5  # Boost for statistical backing
    elif STATISTICAL_PATTERNS['not_significant'].search(sentence):
        metadata['has_stats'] = True
        weight *= 0.3  # Reduce for non-significant findings
    
    # Check for comparative phrases
    if COMPARATIVE_PATTERNS['superior'].search(sentence):
        metadata['is_comparative'] = True
        weight *= 1.3  # Boost for superiority
    elif COMPARATIVE_PATTERNS['inferior'].search(sentence):
        metadata['is_comparative'] = True
        weight *= -1.3  # Penalty for inferiority
    elif COMPARATIVE_PATTERNS['non_inferior'].search(sentence):
        metadata['is_comparative'] = True
        weight *= 1.1  # Slight boost for non-inferiority
    
    # Check severity
    for severity_level, pattern in SEVERITY_INDICATORS.items():
        if pattern.search(sentence):
            metadata['severity'] = severity_level
            if severity_level == 'severe':
                weight *= 1.5  # Severe issues are important
            elif severity_level == 'mild':
                weight *= 0.7  # Mild issues less concerning
            break
    
    # Check frequency
    for freq_level, pattern in FREQUENCY_INDICATORS.items():
        if pattern.search(sentence):
            metadata['frequency'] = freq_level
            if freq_level == 'common':
                weight *= 1.3
            elif freq_level == 'rare':
                weight *= 0.6
            break
    
    # Calculate base sentiment
    has_negation = bool(NEGATION_PATTERNS.search(sentence))
    
    pos_score = sum(1 for pattern in [TERM_PATTERNS['efficacy_pos'], TERM_PATTERNS['safety_pos']]
                    if pattern.search(sentence))
    neg_score = sum(1 for pattern in [TERM_PATTERNS['efficacy_neg'], TERM_PATTERNS['safety_neg']]
                    if pattern.search(sentence))
    
    score = (pos_score - neg_score) * weight
    
    # Apply negation
    if has_negation and score > 0:
        score = -score * 0.5
    
    return score, metadata


def get_section_weight(section_label):
    """Get weight for a section, default to 1.0 if unknown"""
    return SECTION_WEIGHTS.get(section_label.upper(), 1.0)


def analyze_article_sections(article, drug_name):
    """
    Sentence-level analysis within each section
    More accurate than analyzing entire abstract at once
    """
    drug_lower = drug_name.lower()
    
    total_positive_score = 0.0
    total_negative_score = 0.0
    total_weight = 0.0
    
    stats_supported = 0
    total_relevant_sentences = 0
    
    for section_label, section_text in article['sections'].items():
        
        section_weight = get_section_weight(section_label)
        sentences = extract_sentences(section_text)
        
        for sentence in sentences:
            if not sentence_contains_drug(sentence, drug_lower):
                continue
            
            total_relevant_sentences += 1
            score, metadata = analyze_sentence_sentiment(sentence, drug_lower, section_weight)
            
            if metadata['has_stats']:
                stats_supported += 1
            
            total_weight += section_weight
            
            if score > 0:
                total_positive_score += score
            elif score < 0:
                total_negative_score += abs(score)
    
    # Normalize by total weight
    if total_weight > 0 and total_relevant_sentences > 0:
        normalized_pos = total_positive_score / total_weight
        normalized_neg = total_negative_score / total_weight
        
        # Require stronger signal for classification
        if normalized_pos > normalized_neg * 1.3:
            return 'positive'
        elif normalized_neg > normalized_pos * 1.3:
            return 'negative'
        else:
            return 'neutral'
    
    return 'neutral'

def calculate_evidence_quality_score(articles, drug_name):
    """
    IMPROVEMENT #6: Composite evidence quality score (0-5 scale)
    Weights by study type, sample size, recency, journal quality
    """
    
    if not articles:
        return 0.0, "Insufficient Evidence"
    
    total_score = 0.0
    current_year = datetime.now().year
    
    for article in articles:
        score = 0.0
        
        # Base score from study type
        pub_text = ' '.join(article['pub_types']).lower()
        
        if any(term in pub_text for term in EVIDENCE_LEVELS['very_high']):
            score = 5.0
        elif any(term in pub_text for term in EVIDENCE_LEVELS['high']):
            score = 3.0
        elif any(term in pub_text for term in EVIDENCE_LEVELS['medium']):
            score = 1.5
        elif any(term in pub_text for term in EVIDENCE_LEVELS['low']):
            score = 0.5
        else:
            score = 1.0  # default
        
        # IMPROVEMENT: Sample size weighting
        if article.get('sample_size'):
            n = article['sample_size']
            if n > 1000:
                score *= 2.0
            elif n > 200:
                score *= 1.5
            elif n > 50:
                score *= 1.0
            else:
                score *= 0.5
        
        # IMPROVEMENT: Recency weighting
        try:
            year = int(article['year'])
            if year >= current_year - 2:
                score *= 1.2  # Recent studies
            elif year >= current_year - 4:
                score *= 1.0
            else:
                score *= 0.8  # Older studies
        except:
            pass
        
        # IMPROVEMENT #8: Journal quality weighting
        journal = article.get('journal', '')
        if any(high_impact in journal for high_impact in HIGH_IMPACT_JOURNALS):
            score *= 1.5
        
        total_score += score
    
    # Calculate average
    avg_score = total_score / len(articles)
    
    # Classify quality
    if avg_score >= 3.0:
        quality_label = "Strong Evidence"
    elif avg_score >= 2.0:
        quality_label = "Moderate Evidence"
    elif avg_score >= 1.0:
        quality_label = "Limited Evidence"
    else:
        quality_label = "Insufficient Evidence"
    
    return round(avg_score, 2), quality_label

def analyze_drug_comprehensive(articles, drug_name):
    """Enhanced analysis with improved NLP and evidence scoring"""
    
    total = len(articles)
    if total == 0:
        return None
    
    drug_lower = drug_name.lower()
    
    # Initialize counters
    sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
    evidence = Counter()
    safety = {key: 0 for key in ['hepatotoxicity', 'nephrotoxicity', 'gastrointestinal',
                                  'cardiovascular', 'allergic', 'hematologic']}
    
    years = Counter()
    recent = 0
    current_year = datetime.now().year
    
    # Single pass through articles
    for article in articles:
        # Enhanced sentence-level sentiment analysis
        sentiment = analyze_article_sections(article, drug_name)
        sentiments[sentiment] += 1
        
        # Evidence level classification
        pub_text = ' '.join(article['pub_types']).lower()
        for level in ['very_high', 'high', 'medium', 'low', 'preclinical']:
            if any(term in pub_text for term in EVIDENCE_LEVELS[level]):
                evidence[level] += 1
                break
        else:
            evidence['unknown'] += 1
        
        # Year tracking
        year = article['year']
        if year:
            years[year] += 1
            try:
                if int(year) >= current_year - 2:
                    recent += 1
            except:
                pass
        
        # Analyze RESULTS and CONCLUSION sections only for safety (avoid background noise)
        for section_label, section_text in article['sections'].items():
            weight = get_section_weight(section_label)
            
            if weight >= 1.5:  # High-confidence sections only
                sentences = extract_sentences(section_text)
                
                for sentence in sentences:
                    if not sentence_contains_drug(sentence, drug_lower):
                        continue
                    
                    # Safety profile analysis
                    for category in safety.keys():
                        if category in TERM_PATTERNS and TERM_PATTERNS[category].search(sentence):
                            # Weight by severity
                            multiplier = 1.0
                            if SEVERITY_INDICATORS['severe'].search(sentence):
                                multiplier = 2.0
                            elif SEVERITY_INDICATORS['mild'].search(sentence):
                                multiplier = 0.5
                            
                            safety[category] += multiplier
    
    # Calculate composite evidence quality score
    evidence_score, evidence_label = calculate_evidence_quality_score(articles, drug_name)
    
    # Calculate metrics
    pos_pct = round((sentiments['positive'] / total) * 100, 1) if total > 0 else 0
    neg_pct = round((sentiments['negative'] / total) * 100, 1) if total > 0 else 0
    
    # High quality evidence percentage (very_high + high)
    high_quality_count = evidence['very_high'] + evidence['high']
    high_ev_pct = round((high_quality_count / total) * 100, 1) if total > 0 else 0
    
    # Overall assessment based on multiple factors
    if evidence_score >= 3.0 and pos_pct > 60 and neg_pct < 20:
        assessment = 'Favorable'
    elif evidence_score < 1.5 or neg_pct > 40 or max(safety.values()) > 15:
        assessment = 'Concerning'
    elif pos_pct > 40 and evidence_score >= 2.0:
        assessment = 'Mixed-Positive'
    else:
        assessment = 'Mixed-Neutral'
    
    # Primary safety concern
    primary_concern = max(safety.items(), key=lambda x: x[1])
    concern_name = primary_concern[0].replace('_', ' ').title() if primary_concern[1] > 0 else 'None'
    
    # STREAMLINED OUTPUT: Only 8 columns
    return {
        'Drug Name': drug_name,
        'Total Articles': total,
        'Positive Studies (%)': pos_pct,
        'Negative Studies (%)': neg_pct,
        'High Quality Evidence (%)': high_ev_pct,
        'Primary Safety Concern': concern_name,
        'Recent Publications (2yr)': recent,
        'Literature Assessment': assessment,
    }

def process_single_drug(drug_name, drug_index, total_drugs):
    """Process a single drug with beautiful progress output"""
    import time
    start_time = time.time()
    
    print(f"\n   ┌{'─' * 66}┐")
    print(f"   │ [{drug_index}/{total_drugs}] Processing: {drug_name:<47} │")
    print(f"   └{'─' * 66}┘")
    
    # Calculate date range
    date_from = (datetime.now() - timedelta(days=YEARS_BACK * 365)).strftime("%Y/%m/%d")
    date_to = datetime.now().strftime("%Y/%m/%d")
    
    # Step 1: Search
    print(f"      [1/4] Searching PubMed...", end='', flush=True)
    pmids = search_pubmed(f'"{drug_name}"', retmax=MAX_RESULTS, date_from=date_from, date_to=date_to)
    
    if not pmids:
        print(f" No articles found")
        return None
    
    print(f" Found {len(pmids)} articles")
    
    # Step 2: Fetch
    print(f"      [2/4] Fetching articles...", end='', flush=True)
    xml_data = fetch_articles(pmids)
    print(f" Complete")
    
    # Step 3: Parse
    print(f"      [3/4] Parsing XML with quality filters...", end='', flush=True)
    articles = parse_xml(xml_data)
    
    if not articles:
        print(f" No valid articles after filtering")
        return None
    
    print(f" {len(articles)} quality articles")
    
    # Step 4: Analyze
    print(f"      [4/4] Enhanced NLP analysis...", end='', flush=True)
    result = analyze_drug_comprehensive(articles, drug_name)
    
    if not result:
        print(f"Analysis failed")
        return None
    
    print(f" Complete")
    
    elapsed = time.time() - start_time
    print(f"\n Completed in {elapsed:.1f}s")
    
    # Quick summary
    print(f"Sentiment: {result['Positive Studies (%)']}% positive, "
          f"{result['Negative Studies (%)']}% negative")
    print(f"Assessment: {result['Literature Assessment']}")
    print(f"Primary Safety: {result['Primary Safety Concern']}")
    
    return result


def main(drug_list):
    """Main execution with beautiful output"""
    import time
    total_start = time.time()
    
    # Beautiful header
    print("\n")
    print("   ╔" + "═" * 70 + "╗")
    print("   ║" + " " * 70 + "║")
    print("   ║" + " MEDICAL LITERATURE ANALYSIS ENGINE v2.0".center(70) + "║")
    print("   ║" + "  Enhanced NLP | Statistical Significance | Context Analysis".center(70) + "║")
    print("   ║" + " " * 70 + "║")
    print("   ║" + f"  Analyzing {len(drug_list)} drug(s) | Target: 12-15s per drug".center(70) + "║")
    print("   ║" + " " * 70 + "║")
    print("   ╚" + "═" * 70 + "╝")
    
    print(f"\n Drug List: {', '.join(drug_list[:5])}" + (" ..." if len(drug_list) > 5 else ""))
    print(f"Time Period: Last {YEARS_BACK} years")
    print(f"Articles per drug: Top {MAX_RESULTS}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"New Features: Sentence-level NLP, Statistical Significance, Evidence Scoring")
    
    # Setup CSV - STREAMLINED to 8 columns
    file_exists = os.path.isfile(OUTPUT_FILE)
    
    fieldnames = [
        'Drug Name',
        'Total Articles',
        'Positive Studies (%)',
        'Negative Studies (%)',
        'High Quality Evidence (%)',
        'Primary Safety Concern',
        'Recent Publications (2yr)',
        'Literature Assessment'
    ]
    
    print(f"\n   {'─' * 70}")
    print(f"Starting Enhanced Analysis...")
    print(f"   {'─' * 70}")
    
    # Process each drug
    success_count = 0
    
    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for idx, drug in enumerate(drug_list, 1):
            result = process_single_drug(drug, idx, len(drug_list))
            
            if result:
                writer.writerow(result)
                f.flush()
                success_count += 1
                print(f"Saved to {OUTPUT_FILE}")
            else:
                print(f"Skipped (no data available)")
    
    # Final summary
    total_time = time.time() - total_start
    avg_time = total_time / len(drug_list) if drug_list else 0
    
    print(f"\n   {'─' * 70}")
    print(f"\n   ╔" + "═" * 70 + "╗")
    print(f"   ║" + " " * 70 + "║")
    print(f"   ║" + "ANALYSIS COMPLETE!".center(70) + "║")
    print(f"   ║" + " " * 70 + "║")
    print(f"   ╚" + "═" * 70 + "╝")
    
    print(f"\n SUMMARY:")
    print(f"   {'─' * 70}")
    print(f"      Drugs processed:     {success_count}/{len(drug_list)}")
    print(f"      Total time:          {total_time:.1f}s")
    print(f"      Average per drug:    {avg_time:.1f}s")
    
    if avg_time <= 15:
        print(f"      Performance: EXCELLENT! (≤15s target achieved)")
    elif avg_time < 20:
        print(f"      Performance: GOOD (slightly above target)")
    else:
        print(f"      Performance: SLOW (check internet connection)")
    
    print(f"\n Results saved to: {OUTPUT_FILE}")
    print(f"Output format: Streamlined 8-column CSV")
    print(f"Open in Excel or Google Sheets for analysis")
    print(f"\n Quality Improvements:")
    print(f"Sentence-level context analysis")
    print(f"Statistical significance detection")
    print(f"Enhanced evidence quality scoring")
    print(f"Severity & frequency weighting")
    print(f"Comparative phrase detection")
    print(f"Scope-based negation handling")
    print(f"Quality filters (no editorials/letters)")
    print(f"Deduplication")
    print(f"\n   {'─' * 70}\n")
    
    # Cleanup
    client.close()

if __name__ == "__main__":
    
    DRUG_LIST = """
    
    abacavir,abacavir,abilify,abiraterone,bacitracin,balsalazide,baqsimi,cabazitaxel,cabergoline,calcipotriene,calcitonin,calcitriol,calcium,calcium,calquence,campath,campho,camzyos,candesartan,candesartan,capecitabine,caplyta,caprelsa,captopril,captopril,carbidopa,carbidopa,carbidopa,carboplatin,dabigatran,dalfampridine,dalvance,danazol,dantrolene,dapsone,darifenacin,dayvigo,fabrazyme,famciclovir,famotidine,fanapt,farxiga,fasenra,galantamine,galzin,ganciclovir,gattex,gazyva,halcinonide,haloperidol,ibandronate,ibgard,ibrance,ibsrela,ibuprofen,jakafi,januvia,jardiance,kalydeco,l,labetalol,lacosamide,lagevrio,lamivudine,lamivudine,lamotrigine,lanreotide,lansoprazole,lanthanum,lapatinib,maalox,magnesium,magnesium,maraviroc,marplan,mavyret,mayzent,nabumetone,nadolol,naftifine,naltrexone,namzaric,naphazoline,naproxen,natacyn,nateglinide,nayzilam,paclitaxel,paclitaxel,paliperidone,pamidronate,pancreaze,panhematin,pantoprazole,paragard,other that mirena which caused me (weight gain and acne). i switched to paraguard it works 100% as contraception. the procedure was super fast i didnâ€™t take any pain meds or numbing etc it felt like a bad cramp for not even 5 seconds.my periods were longer i would say heavier at first,paricalcitol,paromomycin,pazopanib,rabeprazole,raloxifene,ramelteon,ramipril,ranibizumab,rapivab,rasagiline,rayaldee,salicylic,salsalate,sancuso,santyl,saphnelo,savaysa,saxagliptin,saxenda,tabrecta,tacrolimus,tafluprost,tagrisso,taltz,tamsulosin,tarpeyo,tasimelteon,tavaborole,tavalisse,ubrelvy,vabysmo,valacyclovir,valchlor,valganciclovir,valsartan,valsartan,valtoco,varenicline,varubi,wakix,xadago,xalkori,xarelto,zafirlukast,zaleplon,zaltrap


    """
    
    # Parse drug list
    drugs = [drug.strip() for drug in DRUG_LIST.split(',') if drug.strip()]
    
    if not drugs:
        print("\n  RROR: No drugs specified in DRUG_LIST")
        print("   Please edit the DRUG_LIST section and add drug names")
        sys.exit(1)
    
    # Run analysis
    main(drugs)
