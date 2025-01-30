# encoding:utf-8
from statsbombpy import sb
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsbombpy.api_client")



def get_competition_id_and_season_id(competition_name, competition_gender, season_name):
    '''
    Fetch the competition_id and season_id for a specific competition and season.
    params:
        competition_name (str): The name of the competition.
        competition_gender (str): The gender category of the competition (e.g., 'male', 'female').
        season_name (str): The name of the season.
    returns:
        tuple: A tuple containing the competition_id and season_id.
    '''
    competitions = sb.competitions()
    competition = competitions[(competitions['competition_name'] == competition_name) & 
                                  (competitions['competition_gender'] == competition_gender) & 
                                  (competitions['season_name'] == season_name)]
    if competition.shape[0] == 0:
        raise ValueError('No such competition found. Please check the competition, season and gender name.')
    competition_id = competition['competition_id'].values[0]
    season_id = competition['season_id'].values[0]
    return competition_id, season_id


def get_matches(competition_id, season_id):
    '''
    Fetch the matches for a specific competition and season.
    params:
        competition_id (int): The competition_id.
        season_id (int): The season_id.
    returns:
        DataFrame: A DataFrame containing the matches.
    '''
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    if matches.shape[0] == 0:
        raise ValueError('No such matches found. Please check the competition_id and season_id.')
    return matches


def get_events(match_id):
    '''
    Fetch the events for a specific match.
    params:
        match_id (int): The match_id.
    returns:
        DataFrame: A DataFrame containing the events.
    '''
    events = sb.events(match_id=match_id)
    if events.shape[0] == 0:
        raise ValueError('No such events found. Please check the match_id.')
    return events

