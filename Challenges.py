import pandas as pd
import matplotlib.pyplot as plt
import spacy
import re
import networkx as nx
import unicodedata
from itertools import combinations
from collections import Counter
from spacy.pipeline import EntityRuler

# --- Utility Functions (From Task 1) ---

def load_corpus(filepath="data/climate_articles.csv"):
    return pd.read_csv(filepath)

def preprocess_corpus(df):
    df_copy = df.copy()
    def process_row(row):
        norm_text = unicodedata.normalize('NFC', row['text'])
        return norm_text
    df_copy['processed_text'] = df_copy.apply(process_row, axis=1)
    return df_copy

# --- Tier 1: Temporal Entity Analysis ---

def analyze_temporal_entities(entity_df):
    """Extract and normalize DATE entities to track trends over time."""
    date_entities = entity_df[entity_df['entity_label'] == 'DATE'].copy()
    
    def normalize_date(text):
        # Look for 4-digit years starting with 19 or 20
        year_match = re.search(r'\b(20\d{2}|19\d{2})\b', text)
        return year_match.group(0) if year_match else None

    date_entities['year'] = date_entities['entity_text'].apply(normalize_date)
    date_entities = date_entities.dropna(subset=['year'])
    
    temporal_trends = date_entities.groupby('year').size().reset_index(name='mention_count')
    return temporal_trends

def visualize_temporal_trends(temporal_df, output_path="temporal_trends.png"):
    """Produce a timeline visualization showing entity frequency trends."""
    plt.figure(figsize=(10, 6))
    plt.plot(temporal_df['year'], temporal_df['mention_count'], marker='o', color='orange')
    plt.title('Climate Entity Mention Trends Over Time')
    plt.xlabel('Year')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(trends.head())

# --- Tier 2: Entity Relationship Graph ---

def build_entity_graph(df, nlp):
    """Build a relationship graph based on sentence-level co-occurrence."""
    en_df = df[df['language'] == 'en']
    edges = []

    for doc in nlp.pipe(en_df['processed_text']):
        for sent in doc.sents:
            # Extract entities at sentence level
            sent_entities = sorted(list(set([ent.text for ent in sent.ents])))
            if len(sent_entities) >= 2:
                for pair in combinations(sent_entities, 2):
                    edges.append({'entity_1': pair[0], 'entity_2': pair[1], 'weight': 1})

    if not edges:
        return pd.DataFrame()
        
    edge_df = pd.DataFrame(edges)
    graph_data = edge_df.groupby(['entity_1', 'entity_2']).size().reset_index(name='weight')
    return graph_data.sort_values(by='weight', ascending=False)

def visualize_relationship_graph(graph_data, output_path="entity_graph.png"):
    """Visualize the top 30 entity relationships as a network graph."""
    if graph_data.empty:
        return
        
    G = nx.from_pandas_edgelist(graph_data.head(30), 'entity_1', 'entity_2', ['weight'])
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.5)
    
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', 
            node_size=2500, font_size=10, 
            width=[d['weight']*0.8 for u, v, d in G.edges(data=True)])
            
    plt.title("Entity Relationship Network (Sentence-Level)")
    plt.savefig(output_path)
    plt.close()
    print(graph_data.head())
# --- Tier 3: Domain-Adapted NER with EntityRuler ---

def setup_custom_ner(nlp):
    """Integrate EntityRuler with custom climate patterns into spaCy."""
    # Add entity_ruler before the standard 'ner' component
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    
    patterns = [
        {"label": "CLIMATE_EVENT", "pattern": "COP28"},
        {"label": "CLIMATE_EVENT", "pattern": "COP27"},
        {"label": "POLICY", "pattern": "Paris Agreement"},
        {"label": "REPORT", "pattern": "IPCC AR6"},
        {"label": "THRESHOLD", "pattern": "1.5-degree target"},
        {"label": "THRESHOLD", "pattern": "2-degree target"},
        {"label": "POLICY", "pattern": "Net Zero"},
        {"label": "GAS", "pattern": "Methane"},
        {"label": "GAS", "pattern": "Carbon Dioxide"},
        {"label": "ORG", "pattern": "IPCC"},
        {"label": "REPORT", "pattern": "Sixth Assessment Report"},
        {"label": "POLICY", "pattern": "Kyoto Protocol"},
        {"label": "CLIMATE_EVENT", "pattern": "UN Climate Summit"},
        {"label": "CLIMATE_EVENT", "pattern": "Green Climate Fund"},
        {"label": "REPORT", "pattern": "Global Stocktake"}
    ]
    
    ruler.add_patterns(patterns)
    return nlp

# --- Execution ---

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    
    # Pre-setup for Tier 3
    nlp = setup_custom_ner(nlp)
    
    # Load and prep
    raw_data = load_corpus()
    processed_data = preprocess_corpus(raw_data)
    
    # Run NER to get entities for Tier 1
    en_rows = processed_data[processed_data['language'] == 'en']
    all_entities = []
    for doc, t_id in zip(nlp.pipe(en_rows['processed_text']), en_rows['id']):
        for ent in doc.ents:
            all_entities.append({'text_id': t_id, 'entity_text': ent.text, 'entity_label': ent.label_})
    
    entity_df = pd.DataFrame(all_entities)

    # Execute Tier 1
    trends = analyze_temporal_entities(entity_df)
    visualize_temporal_trends(trends)
    print("Tier 1 complete: Temporal trends saved.")

    # Execute Tier 2
    graph_data = build_entity_graph(processed_data, nlp)
    visualize_relationship_graph(graph_data)
    print("Tier 2 complete: Relationship graph saved.")
    
    print("\nChallenges processed successfully with custom EntityRuler (Tier 3).")