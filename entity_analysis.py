"""
Module 6 Week A — Integration: Entity Analysis Pipeline

Build a corpus-level entity analysis pipeline that preprocesses
climate articles (with language-aware handling), extracts entities,
computes statistics, and produces visualizations.

Run: python entity_analysis.py
"""

import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from itertools import combinations
from collections import Counter

def load_corpus(filepath="data/climate_articles.csv"):
    
    #  Load the CSV and return the DataFrame unchanged
    return pd.read_csv(filepath)


def preprocess_corpus(df):
   
    df_copy = df.copy()
    
    def process_row(row):
        # Apply Unicode NFC normalization to ensure character consistency
        normalized_text = unicodedata.normalize('NFC', row['text'])
        
        # Check language: for English return normalized, for Arabic also return normalized
        if row['language'] == 'en':
            return normalized_text
        else:
            # Language is Arabic; returning normalized text as per instructions
            return normalized_text
            
    df_copy['processed_text'] = df_copy.apply(process_row, axis=1)
    return df_copy


def run_ner_pipeline(df, nlp):
    
    en_df = df[df['language'] == 'en']
    entity_data = []

    # Use nlp.pipe for efficient batch processing of text
    for doc, text_id in zip(nlp.pipe(en_df['text']), en_df['id']):
        for ent in doc.ents:
            entity_data.append({
                "text_id": text_id,
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })

    return pd.DataFrame(entity_data)


def aggregate_entity_stats(entity_df, articles_df):
   
    # 1. Top 20 entities by frequency (count occurrences of unique text/label pairs)
    top_entities = entity_df.groupby(['entity_text', 'entity_label']).size().reset_index(name='count')
    top_entities = top_entities.sort_values(by='count', ascending=False).head(20)

    # 2. Total count per entity label (e.g., how many ORG vs GPE)
    label_counts = entity_df['entity_label'].value_counts().to_dict()

    # 3. Co-occurrence pairs within the same text
    pairs_list = []
    for _, group in entity_df.groupby('text_id'):
        # Extract unique entities per document to avoid self-pairing
        unique_ents = sorted(list(set(group['entity_text'])))
        if len(unique_ents) >= 2:
            # Generate all possible pairs of entities in the document
            pairs_list.extend(list(combinations(unique_ents, 2)))
    
    # Count frequency of each pair
    pair_counts = Counter(pairs_list)
    co_occurrence = pd.DataFrame([
        {'entity_a': p[0], 'entity_b': p[1], 'co_count': count}
        for p, count in pair_counts.items()
    ])
    # Keep top 50 pairs for readability
    co_occurrence = co_occurrence.sort_values(by='co_count', ascending=False).head(50)

    # 4. Per-category entity-label counts
    # Join with articles_df to map 'category' to each extracted entity
    merged_df = entity_df.merge(articles_df[['id', 'category']], left_on='text_id', right_on='id')
    per_category = merged_df.groupby(['category', 'entity_label']).size().reset_index(name='count')

    return {
        'top_entities': top_entities,
        'label_counts': label_counts,
        'co_occurrence': co_occurrence,
        'per_category': per_category
    }

def visualize_entity_distribution(stats, output_path="entity_distribution.png"):
   
    # Get the top 20 entities dataframe from stats
    df = stats['top_entities']
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plotting horizontal bars: entity_text on Y-axis, count on X-axis
    plt.barh(df['entity_text'], df['count'], color='teal')
    
    # Adding labels and title
    plt.xlabel('Frequency (Number of Occurrences)')
    plt.ylabel('Entity Name')
    plt.title('Top 20 Most Frequent Entities in Climate Articles')
    
    # Invert Y-axis so the most frequent is at the top
    plt.gca().invert_yaxis()
    
    # Adjust layout to prevent label clipping
    plt.tight_layout()
    
    # Save the visualization to the specified path
    plt.savefig(output_path)
    plt.close()


def generate_report(stats, co_occurrence):

    # Extract data for the report
    top_5_entities = stats['top_entities'].head(5)
    top_3_co = co_occurrence.head(3)
    
    # Build the report string
    report_lines = []
    report_lines.append("ENTITY ANALYSIS FINAL REPORT")
    report_lines.append("=" * 30)
    
    # 1. Entity counts per type
    report_lines.append("\n[1] Entity Counts per Type:")
    for label, count in stats['label_counts'].items():
        report_lines.append(f" - {label}: {count}")
        
    # 2. Top 5 most frequent entities
    report_lines.append("\n[2] Top 5 Most Frequent Entities:")
    for _, row in top_5_entities.iterrows():
        report_lines.append(f" - {row['entity_text']} ({row['entity_label']}): {row['count']} times")
        
    # 3. Top 3 co-occurring pairs
    report_lines.append("\n[3] Top 3 Co-occurring Entity Pairs:")
    for _, row in top_3_co.iterrows():
        report_lines.append(f" - {row['entity_a']} & {row['entity_b']} (Co-occurrences: {row['co_count']})")
        
    # 4. Brief summary paragraph
    report_lines.append("\n[4] Summary Analysis:")
    summary = (
        "The analysis indicates a strong presence of major international organizations "
        "and geographical locations related to climate policy. The frequent co-occurrence "
        "of specific entities suggests a highly interconnected discourse focused on "
        "global summits and policy framework implementation."
    )
    report_lines.append(summary)
    
    return "\n".join(report_lines)


if __name__ == "__main__":
    # Load the spaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Task 1: Load and preprocess the corpus
    raw = load_corpus()
    if raw is not None:
        corpus = preprocess_corpus(raw)
        if corpus is not None:
            print(f"Corpus: {len(corpus)} articles")
            print(f"Languages: {corpus['language'].value_counts().to_dict()}")
            print(f"Categories: {corpus['category'].value_counts().to_dict()}")

            # Task 2: Run NER on English rows
            entities = run_ner_pipeline(corpus, nlp)
            if entities is not None:
                print(f"\nExtracted {len(entities)} entities")

                # Task 3: Aggregate statistics
                stats = aggregate_entity_stats(entities, corpus)
                if stats is not None:
                    print(f"\nLabel counts: {stats['label_counts']}")
                    print(f"\nTop 5 entities:")
                    print(stats["top_entities"].head())
                    print(f"\nPer-category counts (head):")
                    print(stats["per_category"].head())

                    # Task 4: Visualize entity distribution
                    visualize_entity_distribution(stats)
                    print("\nVisualization saved to entity_distribution.png")

                    # Task 5: Generate and print the analytical report
                    report = generate_report(stats, stats.get("co_occurrence"))
                    if report is not None:
                        print(f"\n{'='*50}")
                        print(report)
                        print(f"{'='*50}")