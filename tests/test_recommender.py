import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

from recommender import FEATURES, build_model, clean_data, get_cover, recommend


class RecommenderTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame(rng.random((15, len(FEATURES))), columns=FEATURES)
        data['track_id'] = [str(i).zfill(22) for i in range(15)]
        data['track_name'] = [f'Song {i}' for i in range(15)]
        data['artists'] = [f'Artist {i}' for i in range(15)]
        data['track_genre'] = 'pop'
        data['popularity'] = range(15)
        self.data = data

    def test_genres_are_preserved_and_missing_values_imputed(self):
        duplicate = self.data.iloc[[0]].copy()
        duplicate['track_genre'] = 'rock'
        songs = clean_data(pd.concat([self.data, duplicate]))
        self.assertEqual(len(songs), 15)
        self.assertEqual(songs.iloc[0]['genres'], 'pop|rock')
        songs.loc[0, 'energy'] = np.nan
        self.assertTrue(np.isfinite(build_model(songs)[2]).all())

    def test_six_unique_recommendations_exclude_query_and_same_song(self):
        self.data.loc[1, ['track_name', 'artists']] = self.data.loc[0, ['track_name', 'artists']]
        songs = clean_data(self.data)
        for metric in ['cosine', 'euclidean', 'popularity']:
            results = recommend(songs.iloc[0], songs, build_model(songs), metric)
            self.assertEqual(len(results), 6)
            self.assertEqual(results['song_key'].nunique(), 6)
            self.assertNotIn(songs.iloc[0]['song_key'], results['song_key'].tolist())
            self.assertTrue(results['score'].is_monotonic_decreasing)

    def test_heldout_query_does_not_change_scaler(self):
        songs = clean_data(self.data)
        model = build_model(songs.iloc[:10])
        mean = model[0].mean_.copy()
        recommend(songs.iloc[14], songs.iloc[:10], model)
        np.testing.assert_equal(mean, model[0].mean_)

    @patch('recommender.requests.get')
    def test_artwork_failure_and_success(self, get):
        get.side_effect = requests.Timeout()
        self.assertIsNone(get_cover('0' * 22))
        get.side_effect = None
        get.return_value = Mock(json=lambda: {'thumbnail_url': 'https://i.scdn.co/image/example'})
        self.assertEqual(get_cover('0' * 22), 'https://i.scdn.co/image/example')


if __name__ == '__main__':
    unittest.main()
