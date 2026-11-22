# Song Recommender

A beginner-friendly Streamlit app that recommends six similar songs using Spotify audio features. Search by title or artist, choose a track, and explore its recommendations with album covers and Spotify links.

## Run locally

Use Python 3.10 or newer.

```bash
git clone https://github.com/raoankit72005/recommender-system.git
cd recommender-system
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app downloads `dataset.csv` directly from [Hugging Face](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) on the first run and saves it in `data/`. No manual CSV upload or Hugging Face token is needed. The dataset is not committed to GitHub. Delete `data/dataset.csv` to download a fresh copy.

## How it works

1. Drop invalid identifiers and missing song names/artists.
2. Combine genre labels before removing repeated track IDs.
3. Convert nine audio features to numbers and treat invalid values as missing.
4. Fill missing values with medians and standardize features.
5. Compare the selected song with the catalog using cosine similarity.
6. Remove the selected song and repeated title/artist combinations, then return six results.

The features are danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, and tempo. Popularity and genre are not inputs to the audio model. The score is a similarity score, not a probability. The app calculates one query against the catalog instead of storing an enormous all-pairs matrix.

## Posters and playback

Album thumbnails come from [Spotify oEmbed](https://developer.spotify.com/documentation/embeds/reference/oembed) using the exact track ID. This endpoint does not require a client ID or secret. Requests have a timeout, and missing artwork shows a placeholder. Thumbnail URLs are cached briefly in Streamlit; covers link back to Spotify through each card's button.

The selected track has an official Spotify embedded player. Press play in that player. Spotify controls playback availability, previews, login requirements, and regional restrictions. The dataset contains no audio files. This app bases recommendations on the track you select; it does not monitor playback in your external Spotify app. Click “Recommend from this song” on a result to make it the selected track.

## Evaluation

```bash
python evaluate.py --queries 100
```

This creates `results/evaluation.json`. The script uses approximately 80/10/10 train/validation/test splits grouped by normalized title and artist, seed 42. Duplicate track IDs are removed before splitting. Exact title/artist variants stay together, but remixes with different names may still cross partitions.

Median imputation and scaling are fitted only on training tracks. Training tracks are the candidate catalog. Validation and test tracks are unseen queries. Popularity, cosine, and Euclidean methods are compared on identical validation queries. The better audio metric by validation NDCG@6 is evaluated on test queries. The app starts with cosine; evaluation does not silently change it. To use Euclidean after reviewing results, pass `metric='euclidean'` to `recommend` in `app.py`.

Reported metrics:

- Precision@6: fraction of six results sharing at least one genre with the query.
- NDCG@6: ranking quality under the same binary genre-overlap definition.
- Distinct primary artists@6: an artist-diversity indicator.
- Catalog coverage: fraction of candidate track IDs returned across sampled queries.
- Mean latency: recommendation computation time, excluding artwork and network time.

These are genre-based proxies, not measurements of user preference. The dataset has no listening history or user relevance judgments. Coverage depends on the number of evaluated queries. For a stronger evaluation, ask listeners to rate recommendations for 30–50 varied songs without seeing which method generated them. No benchmark scores are claimed before running the script.

## Files

| File | Purpose |
| --- | --- |
| `recommender.py` | Download, preprocessing, similarity, and album artwork |
| `app.py` | Simple search, player, and six-card Streamlit interface |
| `evaluate.py` | Reproducible comparison and held-out evaluation |
| `tests/test_recommender.py` | Checks for duplicates, missing values, ranking, and artwork failure |

```bash
python -m unittest discover -s tests -v
```

The code uses plain functions, pandas, and scikit-learn without a complex pipeline. Read `recommender.py` first, followed by `app.py` and `evaluate.py`.

## Data attribution

Dataset: [maharshipandya/spotify-tracks-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset). Refer to its dataset card for provenance and license. Spotify artwork and playback are supplied through Spotify and remain subject to Spotify's terms.
