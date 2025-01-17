# encoding:utf-8
from src.fetch_data import get_events
import pandas as pd
import numpy as np



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
        match_events_sorted_by_index_df = match_events.sort_values(by=["index","type"])
        # convertimos toda la información del partido en métricas
        match_metrics = _process_match(match_events_sorted_by_index_df, home_team, away_team, winning_team)
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
        # estadísticas de generales del partido
        ## tiros
        "total_shots_home": _num_event_type(events_df, home_team, 'Shot'),
        "total_shots_away": _num_event_type(events_df, away_team, 'Shot'),
        "shots_on_target_ratio_home": _ratio_shots_on_target(events_df, home_team),
        "shots_on_target_ratio_away": _ratio_shots_on_target(events_df, away_team),
        "average_shots_on_target_distance_home": _average_distance_to_goal_of_shots_on_target(events_df, home_team),
        "average_shots_on_target_distance_away": _average_distance_to_goal_of_shots_on_target(events_df, away_team),
        "shots_high_xG_home": _num_shots_high_xG(events_df, home_team),
        "shots_high_xG_away": _num_shots_high_xG(events_df, away_team),
        "shots_inside_area_home": _num_shots_inside_area(events_df, home_team),
        "shots_inside_area_away": _num_shots_inside_area(events_df, away_team),
        "shots_inside_area_ratio_home": _ratio_shots_inside_area(events_df, home_team),
        "shots_inside_area_ratio_away": _ratio_shots_inside_area(events_df, away_team),
        "shots_foot_home": _num_shots_with_body_part(events_df, home_team, "Foot"),
        "shots_foot_away": _num_shots_with_body_part(events_df, away_team, "Foot"),
        "shots_head_home": _num_shots_with_body_part(events_df, home_team, "Head"),
        "shots_head_away": _num_shots_with_body_part(events_df, away_team, "Head"),
        "shots_other_home": _num_shots_with_body_part(events_df, home_team, "Other"),
        "shots_other_away": _num_shots_with_body_part(events_df, away_team, "Other"),
        ## pases
        "total_passes_home": _num_event_type(events_df, home_team, 'Pass'),
        "total_passes_away": _num_event_type(events_df, away_team, 'Pass'),
        "pass_success_ratio_home": _ratio_sucess_passes(events_df, home_team),
        "pass_success_ratio_away": _ratio_sucess_passes(events_df, away_team),
        "key_passes_home": _num_key_passes(events_df, home_team),
        "key_passes_away": _num_key_passes(events_df, away_team),
        "passes_needed_to_make_a_shoot_home": _num_passes_needed_to_make_a_shoot(events_df, home_team),
        "passes_needed_to_make_a_shoot_away": _num_passes_needed_to_make_a_shoot(events_df, away_team),
        "crosses_home": _num_crosses(events_df, home_team),
        "crosses_away": _num_crosses(events_df, away_team),
        "cross_success_ratio_home": _ratio_success_crosses(events_df, home_team),
        "cross_success_ratio_away": _ratio_success_crosses(events_df, away_team),
        "corners_home": _num_corners(events_df, home_team),
        "corners_away": _num_corners(events_df, away_team),
        # equipo ganador
        "winning_team": winning_team,
    }
    return metrics


def _num_event_type(events_df, team, event_type):
    '''
    Calculate the number of the event type selected for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        int: The number of shots.
    '''
    return events_df[(events_df['team'] == team) & (events_df['type'] == event_type)].shape[0]


def _ratio_shots_on_target(events_df, team):
    '''
    Calculate the ratio of shots on target for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        float: The ratio of shots on target.
    '''
    shots = _num_event_type(events_df, team, 'Shot')
    shots_on_target = _shots_on_target_df(events_df, team).shape[0]
    return shots_on_target / shots if shots > 0 else 0.0


def _shots_on_target_df(events_df, team):
    '''
    Get the shots on target DataFrame for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:    
        DataFrame: A DataFrame containing the shots on target
    '''
    # dataframe de los disparos a puerta claros
    shots_on_target = events_df[(events_df['team'] == team) & (events_df['type'] == 'Shot') &
                                (events_df['shot_outcome'].isin(["Goal", "Saved", "Saved To Post"]))
                                ].copy()
    # dataframes para ver si un tiro bloqueado iba a puerta o no mirando si el bloqueo es un save_block
    blocked_shots = events_df[(events_df['type'] == 'Shot') & (events_df['team'] == team) &
                              (events_df['shot_outcome'] == "Blocked") &
                              (events_df['related_events'].notnull())  # para asegurar que tiene related_events
                              ].copy()
    blocked_shots['related_blocks'] = blocked_shots['related_events'].apply(
        lambda rel_events: [event_id for event_id in rel_events 
            if event_id in events_df['id'].values and 
            events_df[events_df['id'] == event_id].iloc[0]['type'] == 'Block' and 
            'block_save_block' in events_df.columns and
            events_df[events_df['id'] == event_id].iloc[0]['block_save_block'] == True
        ]
    )
    valid_blocked_shots = blocked_shots[blocked_shots['related_blocks'].str.len() > 0]
    # concatenamos los disparos a puerta claros con los disparos bloqueados que iban a puerta
    total_shots_on_target = pd.concat([shots_on_target, valid_blocked_shots], ignore_index=True)
    return total_shots_on_target


def _average_distance_to_goal_of_shots_on_target(events_df, team):
    '''
    Calculate the average distance to goal of shots on target for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        float: The average distance to goal of shots on target.
    '''
    shots_on_target = _shots_on_target_df(events_df, team).copy()
    shots_on_target = shots_on_target[shots_on_target['location'].notnull()]
    if(shots_on_target.shape[0] > 0):   # si hay tiros a puerta
        # calculamos la distancia de cada disparo al centro de la portería
        goal_x, goal_y = 120, 40  # coordenadas del centro de la portería
        shots_on_target['distance_to_goal'] = np.sqrt(
            (goal_x - shots_on_target['location'].str[0])**2 + 
            (goal_y - shots_on_target['location'].str[1])**2
        )
        # calculamos la media de las distancias
        average_distance = shots_on_target['distance_to_goal'].mean()
    else:   # si no hay tiros a puerta
        average_distance = 120.0  # la long. del campo de fútbol de statsbomb
    return average_distance


def _num_shots_high_xG(events_df, team):
    '''
    Calculate the number of shots with a high xG for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        int: The number of shots with a high xG.
    '''
    return events_df[(events_df['team'] == team) & (events_df['type'] == 'Shot') & 
                     (events_df['shot_statsbomb_xg'] > 0.2)].shape[0]


def _num_shots_inside_area(events_df, team):
    '''
    Calculate the number of shots inside the area for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        int: The number of shots inside the area.
    '''
    return events_df[(events_df['team'] == team) & (events_df['type'] == 'Shot') &
                     (events_df['location'].notnull()) &
                     (events_df['location'].apply(lambda loc: isinstance(loc, list) and 102 <= loc[0] <= 120 and 18 <= loc[1] <= 62))].shape[0]
                     

def _ratio_shots_inside_area(events_df, team):
    '''
    Calculate the ratio of shots inside the area for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        float: The ratio of shots inside the area.
    '''
    shots = _num_event_type(events_df, team, 'Shot')
    shots_inside_area = _num_shots_inside_area(events_df, team)
    return shots_inside_area / shots if shots > 0 else 0.0


def _num_shots_with_body_part(events_df, team, body_part):
    '''
    Calculate the number of shots with a specific body part for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
        body_part (str): The body part.
    returns:
        int: The number of shots with a specific body part.
    '''
    num_shots = 0
    if body_part == "Foot" :
        num_shots = events_df[(events_df['team'] == team) & (events_df['type'] == 'Shot') & 
                              (events_df['shot_body_part'].notnull()) &
                              (events_df['shot_body_part'].isin(["Right Foot","Left Foot"]))].shape[0]
    else:
        num_shots = events_df[(events_df['team'] == team) & (events_df['type'] == 'Shot') & 
                              (events_df['shot_body_part'].notnull()) &
                              (events_df['shot_body_part'] == body_part)].shape[0]
    return num_shots


def _ratio_sucess_passes(events_df, team):
    '''
    Calculate the ratio of successful passes for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        float: The ratio of successful passes.
    '''
    total_passes = _num_event_type(events_df, team, 'Pass')
    successful_passes = events_df[(events_df['team'] == team) & (events_df['type'] == 'Pass') &
                                   (events_df['pass_outcome'].isnull())].shape[0]
    return successful_passes / total_passes if total_passes > 0 else 0.0


def _num_key_passes(events_df, team):
    '''
    Calculate the number of key passes for a specific team.
    We consider a key pass to be one that assists a shot.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        int: The number of key passes.
    '''
    num_key_passes = 0
    if 'pass_assisted_shot_id' in events_df.columns:
        num_key_passes = events_df[(events_df['team'] == team) & (events_df['type'] == 'Pass') &
                                   (events_df['pass_outcome'].isnull()) &                       # pase exitoso           
                                   (events_df['pass_assisted_shot_id'].notnull())].shape[0]     # pase asistente
    return num_key_passes
    
def _num_passes_needed_to_make_a_shoot(events_df, team):
    '''
    Calculate the number of passes needed to make a shoot for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        int: The number of passes needed to make a shoot.
    '''
    num_shots = _num_event_type(events_df, team, 'Shot')
    num_passes = _num_event_type(events_df, team, 'Pass')
    return num_passes / num_shots if num_shots > 0 else 0.0


def _num_crosses(events_df, team):
    '''
    Calculate the number of crosses for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        int: The number of crosses.
    '''
    num_crosses = 0
    if 'pass_cross' in events_df.columns:
        num_crosses = events_df[(events_df['team'] == team) & (events_df['type'] == 'Pass') &
                                (events_df['pass_cross'] == True)].shape[0]
    return num_crosses


def _ratio_success_crosses(events_df, team):
    '''
    Calculate the ratio of successful crosses for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        float: The ratio of successful crosses.
    '''
    total_crosses = _num_crosses(events_df, team)
    successful_crosses = events_df[(events_df['team'] == team) & (events_df['type'] == 'Pass') &
                                   (events_df['pass_cross'] == True) &
                                   (events_df['pass_outcome'].isnull())].shape[0]
    return successful_crosses / total_crosses if total_crosses > 0 else 0.0


def _num_corners(events_df, team):
    '''
    Calculate the number of corners for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        int: The number of corners.
    '''
    return events_df[(events_df['team'] == team) & (events_df['type'] == 'Pass') &
                     (events_df['pass_type'] == 'Corner')].shape[0]

