from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

DATA_URL = 'https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv'
DATA_PATH = Path(__file__).parent / 'data' / 'dataset.csv'
FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo'
]


def load_data():
    if not DATA_PATH.exists():
        response = requests.get(DATA_URL, timeout=120)
        response.raise_for_status()
        DATA_PATH.parent.mkdir(exist_ok=True)
        temporary_path = DATA_PATH.with_suffix('.tmp')
        temporary_path.write_bytes(response.content)
        pd.read_csv(temporary_path, nrows=1)[FEATURES + ['track_id']]
        temporary_path.replace(DATA_PATH)
    return pd.read_csv(DATA_PATH)


def clean_data(data):
    songs = data.copy()
    songs = songs.dropna(subset=['track_id', 'track_name', 'artists'])
    for column in ['track_id', 'track_name', 'artists']:
        songs[column] = songs[column].astype(str).str.strip()
        songs = songs[songs[column] != '']
    songs = songs[songs['track_id'].str.fullmatch(r'[A-Za-z0-9]{22}')]
    songs['track_genre'] = songs['track_genre'].fillna('')
    genres = songs.groupby('track_id')['track_genre'].agg(
        lambda values: '|'.join(sorted(set(values) - {''}))
    )
    songs = songs.drop_duplicates('track_id').copy()
    songs['genres'] = songs['track_id'].map(genres)
    songs['song_key'] = (
        songs['track_name'].str.casefold().str.replace(r'\s+', ' ', regex=True)
        + ' | ' + songs['artists'].str.casefold().str.replace(r'\s+', ' ', regex=True)
    )
    for feature in FEATURES:
        songs[feature] = pd.to_numeric(songs[feature], errors='coerce')
    songs[FEATURES] = songs[FEATURES].replace([np.inf, -np.inf], np.nan)
    for feature in FEATURES:
        if feature not in ['loudness', 'tempo']:
            songs.loc[~songs[feature].between(0, 1), feature] = np.nan
    songs.loc[songs['tempo'] <= 0, 'tempo'] = np.nan
    songs['popularity'] = pd.to_numeric(songs['popularity'], errors='coerce').fillna(0)
    return songs.reset_index(drop=True)


def build_model(songs):
    medians = songs[FEATURES].median().fillna(0)
    values = songs[FEATURES].fillna(medians)
    scaler = StandardScaler()
    vectors = scaler.fit_transform(values)
    return scaler, medians, vectors


def recommend(query, songs, model, metric='cosine', count=6):
    scaler, medians, vectors = model
    values = pd.DataFrame([query[FEATURES]])
    query_vector = scaler.transform(values.fillna(medians))
    if metric == 'cosine':
        scores = cosine_similarity(query_vector, vectors)[0]
    elif metric == 'euclidean':
        scores = -euclidean_distances(query_vector, vectors)[0]
    elif metric == 'popularity':
        scores = songs['popularity'].to_numpy()
    else:
        raise ValueError('Choose cosine, euclidean, or popularity.')
    ranked = songs.copy()
    ranked['score'] = scores
    ranked = ranked[ranked['track_id'] != query['track_id']]
    ranked = ranked[ranked['song_key'] != query['song_key']]
    ranked = ranked.sort_values('score', ascending=False, kind='stable')
    return ranked.drop_duplicates('song_key').head(count).reset_index(drop=True)


def get_cover(track_id):
    try:
        response = requests.get(
            'https://open.spotify.com/oembed',
            params={'url': f'https://open.spotify.com/track/{track_id}'},
            timeout=5
        )
        response.raise_for_status()
        url = response.json().get('thumbnail_url')
        return url if isinstance(url, str) and url.startswith('https://') else None
    except (requests.RequestException, ValueError, AttributeError):
        return None
