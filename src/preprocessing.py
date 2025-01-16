# encoding:utf-8
import pandas as pd
from src.fetch_data import get_events



def process_all_matches(matches_df):
    '''
    Process (obtain all relevant data) all matches in the DataFrame.
    params:
        matches_df (DataFrame): A DataFrame containing the matches.
    returns:
        DataFrame: A DataFrame containing the processed matches.
    '''
    all_matches_metrics = []
    
    for _, match in matches_df.iterrows():
        # obtenemos toda la información relativa al partido
        match_id = match['match_id']
        home_team = match['home_team']
        away_team = match['away_team']
        winning_team ='home_team' if match['home_score'] > match['away_score'] else 'draw' if match['away_score'] == match['home_score'] else 'away_team'
        match_events = get_events(match_id)
        # convertimos toda la información del partido en métricas
        match_metrics = _process_match(match_events, home_team, away_team, winning_team)
        all_matches_metrics.append(match_metrics)
    
    return pd.DataFrame(all_matches_metrics)


def _process_match(events_df, home_team, away_team, winning_team):
    '''
    Process (obtain all relevant data) a match.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        home_team (str): The home team name.
        away_team (str): The away team name.
        winning_team (str): The winning team (home_team, away_team, draw).
    returns:
        dict: A dictionary containing the processed match.    
    '''
    metrics = {
        "total_shots_home": _calculate_num_event_type(events_df, home_team, 'Shot'),
        "total_shots_away": _calculate_num_event_type(events_df, away_team, 'Shot'),
        "winning_team": winning_team,
    }

    return metrics


def _calculate_num_event_type(events_df, team, event_type):
    '''
    Calculate the number of the event type selected for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        int: The number of shots.
    '''
    return events_df[(events_df['team'] == team) & (events_df['type'] == event_type)].shape[0]

