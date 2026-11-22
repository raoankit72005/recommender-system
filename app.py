import streamlit as st
import streamlit.components.v1 as components

from recommender import load_data, clean_data, build_model, recommend, get_cover

st.set_page_config(page_title='Song Recommender', page_icon='🎵', layout='wide')


@st.cache_resource
def prepare_catalog():
    songs = clean_data(load_data())
    return songs, build_model(songs)


@st.cache_data(ttl=3600, max_entries=500)
def cover_for(track_id):
    return get_cover(track_id)


def choose_song(track_id):
    st.session_state['current_track'] = track_id


st.title('🎵 Song Recommender')
st.write('Pick a song and discover six tracks with a similar sound.')

try:
    with st.spinner('Loading songs from Hugging Face…'):
        songs, model = prepare_catalog()
except Exception:
    st.error('Could not load the dataset. Check your internet connection and try again.')
    st.stop()

search = st.text_input('Search by song name or artist', placeholder='For example: Arijit Singh')
if search.strip():
    matches = songs[
        songs['track_name'].str.contains(search.strip(), case=False, regex=False)
        | songs['artists'].str.contains(search.strip(), case=False, regex=False)
    ].head(100)
else:
    matches = songs.sort_values('popularity', ascending=False).head(50)

if matches.empty:
    st.info('No matching songs. Try a different name or artist.')
else:
    labels = {
        row.track_id: f'{row.track_name} — {row.artists} [{row.track_id[-6:]}]'
        for row in matches.itertuples()
    }
    selected = st.selectbox('Choose a song', list(labels), format_func=labels.get)
    st.button('Find 6 similar songs', type='primary', on_click=choose_song, args=(selected,))

if 'current_track' in st.session_state:
    current = songs[songs['track_id'] == st.session_state['current_track']].iloc[0]
    st.subheader('Selected track')
    st.write(f"{current['track_name']} — {current['artists']}")
    components.iframe(
        f"https://open.spotify.com/embed/track/{current['track_id']}", height=152
    )
    st.caption('Press play above. Playback availability depends on Spotify and your region/account.')
    recommendations = recommend(current, songs, model)
    st.subheader('6 similar songs')
    for start in range(0, len(recommendations), 3):
        columns = st.columns(3)
        for column, (_, song) in zip(columns, recommendations.iloc[start:start + 3].iterrows()):
            with column:
                with st.container(border=True):
                    cover = cover_for(song['track_id'])
                    if cover:
                        st.image(cover, width=220)
                    else:
                        st.info('🎵 Album cover unavailable')
                    st.write(song['track_name'])
                    st.caption(song['artists'])
                    st.caption(f"Similarity score: {song['score']:.3f}")
                    st.link_button('Open in Spotify', f"https://open.spotify.com/track/{song['track_id']}")
                    st.button('Recommend from this song', key=song['track_id'],
                              on_click=choose_song, args=(song['track_id'],))
    st.caption('Artwork provided by Spotify. Similarity is based on audio features, not a probability you will like a song.')
else:
    st.info('Select a song above to get started.')
