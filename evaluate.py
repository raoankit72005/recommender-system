import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from recommender import load_data, clean_data, build_model, recommend


def evaluate(queries, catalog, model, metric):
    precision, ndcg, diversity, latency = [], [], [], []
    recommended_ids = set()
    for _, query in queries.iterrows():
        start = perf_counter()
        results = recommend(query, catalog, model, metric)
        latency.append((perf_counter() - start) * 1000)
        query_genres = set(query['genres'].split('|')) - {''}
        relevant = np.array([
            bool(query_genres & (set(genres.split('|')) - {''}))
            for genres in results['genres']
        ], dtype=float)
        eligible = catalog[catalog['song_key'] != query['song_key']]
        relevant_keys = eligible.loc[
            eligible['genres'].map(lambda value: bool(query_genres & (set(value.split('|')) - {''}))),
            'song_key'
        ].nunique()
        discounts = np.log2(np.arange(2, len(results) + 2))
        dcg = np.sum(relevant / discounts)
        ideal_count = min(6, relevant_keys)
        ideal_dcg = np.sum(1 / np.log2(np.arange(2, ideal_count + 2)))
        precision.append(relevant.sum() / 6)
        ndcg.append(dcg / ideal_dcg if ideal_dcg else 0)
        diversity.append(results['artists'].str.split(';').str[0].nunique())
        recommended_ids.update(results['track_id'])
    return {
        'precision_at_6': float(np.mean(precision)),
        'ndcg_at_6': float(np.mean(ndcg)),
        'distinct_primary_artists_at_6': float(np.mean(diversity)),
        'catalog_coverage': len(recommended_ids) / len(catalog),
        'mean_latency_ms': float(np.mean(latency)),
        'query_count': len(queries)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=int, default=100)
    args = parser.parse_args()
    if args.queries < 1:
        parser.error('--queries must be positive')
    songs = clean_data(load_data())
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_ids, heldout_ids = next(splitter.split(songs, groups=songs['song_key']))
    train = songs.iloc[train_ids].reset_index(drop=True)
    heldout = songs.iloc[heldout_ids].reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_ids, test_ids = next(splitter.split(heldout, groups=heldout['song_key']))
    validation = heldout.iloc[val_ids].sample(n=min(args.queries, len(val_ids)), random_state=42)
    test = heldout.iloc[test_ids].sample(n=min(args.queries, len(test_ids)), random_state=42)
    model = build_model(train)
    validation_results = {}
    for metric in ['popularity', 'cosine', 'euclidean']:
        validation_results[metric] = evaluate(validation, train, model, metric)
        print(metric, validation_results[metric], flush=True)
    best_metric = max(['cosine', 'euclidean'], key=lambda name: validation_results[name]['ndcg_at_6'])
    report = {
        'description': 'Genre-overlap proxy evaluation; not user satisfaction or recommendation accuracy.',
        'seed': 42,
        'split_rows': {'train': len(train), 'validation': len(val_ids), 'test': len(test_ids)},
        'validation': validation_results,
        'selected_metric': best_metric,
        'test': evaluate(test, train, model, best_metric)
    }
    Path('results').mkdir(exist_ok=True)
    Path('results/evaluation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
