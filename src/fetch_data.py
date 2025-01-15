# encoding:utf-8
from statsbombpy import sb



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
    competition_id = competition['competition_id'].values[0]
    season_id = competition['season_id'].values[0]
    return (competition_id, season_id)

    