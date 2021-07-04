import argparse
import os
import re
import time
import pandas as pd
import lyricsgenius
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


def lyrics_by_album(artist):
    # REQUIRES ENVIRONMENT VARIABLES SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
    artist_results = spotify.search(q='artist:' + artist, type='artist', limit=1)
    uri = artist_results['artists']['items'][0]['uri']
    
    album_results = spotify.artist_albums(uri, album_type='album')
    albums = album_results['items']
    while album_results['next']:
        album_results = spotify.next(album_results)
        albums.extend(album_results['items'])

    album_names = set()
    for album in albums:
        album_name_no_brackets = re.sub('\s*\(.*\)$', '', album['name'])
        album_names.add(album_name_no_brackets)

    for album_name in album_names:
        album = genius.search_album(album_name, artist)
        filename = './discography_lyrics/{}_{}'.format(artist.replace(" ", ""), 
                                                    album_name.replace(" ", ""))
        album.save_lyrics(filename=filename, overwrite=True)

def lyrics_by_song(artist, title):
    succ = False
    while not succ:
        try:
            song = genius.search_song(title, artist)
            if not song:
                print("Could not find specified song. Check spelling?")
                return
            filename = './song_lyrics/{}_{}'.format(artist.replace(" ", ""), 
                                                    title.replace(" ", ""))
            song.save_lyrics(filename=filename, overwrite=True)
            succ = True
        except Exception as e:
            print(e)
            time.sleep(70)

        time.sleep(0.1)
        
def moodylyrics(path):
    df = pd.read_excel(path)
    search_terms = list(zip(df['Artist'].values, df['Title'].values, df['Mood'].values))
    
    for artist, title, mood in search_terms[1427:]:
        succ = False
        while not succ:
            try:
                song = genius.search_song(title, artist)
                if not song:
                    print("Could not find specified song. Check spelling?")
                    succ = True
                    continue
                filename = './ml_lyrics/{}_{}_{}'.format(artist.replace(" ", ""), 
                                                        title.replace(" ", ""), mood)
                song.save_lyrics(filename=filename, overwrite=True)
                succ = True
            except Exception as e:
                print(e)
                time.sleep(70)

            time.sleep(0.1)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, help='Specify either MoodyLyrics or song or album search terms [ml/song/album]', dest='type')
    
    # MoodyLyrics
    parser_ml = subparsers.add_parser('ml')
    parser_ml.add_argument('--dir', type=str, default="./", help='filepath to moodylrics')

    # Individual song
    parser_song = subparsers.add_parser('song')
    parser_song.add_argument('artist', type=str, help='Artist name')
    parser_song.add_argument('title', type=str, help='Title of track')

    # Album
    parser_song = subparsers.add_parser('album')
    parser_song.add_argument('artist', type=str, help='Artist name')

    args = parser.parse_args()
    print(args)
    # REQUIRES ENVIRONMENT VARIABLE GENIUS_ACCESS_TOKEN
    token = os.environ.get('GENIUS_ACCESS_TOKEN')
    genius = lyricsgenius.Genius(token)

    if args.type == 'ml':
        os.mkdir('./ml_lyrics')
        moodylyrics('./ml_balanced.xlsx')
    elif args.type == 'song':
        os.mkdir('./song_lyrics')
        lyrics_by_song(args.artist, args.title)
    elif args.type == 'album':
        os.mkdir('./discography_lyrics')
        lyrics_by_album(args.artist)

    


    
    
