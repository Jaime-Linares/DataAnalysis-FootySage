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
    possession_percentage = _possession_percentage(events_df, home_team, away_team)
    possession_percentage_home = possession_percentage[0]
    possession_percentage_away = possession_percentage[1]
    metrics = {
        # estadísticas generales del partido
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
        ## defensa
        "interceptions_won_home": _num_interceptions_won(events_df, home_team),
        "interceptions_won_away": _num_interceptions_won(events_df, away_team),
        "recoveries_home": _num_recoveries(events_df, home_team),
        "recoveries_away": _num_recoveries(events_df, away_team), 
        "blocks_home": _num_event_type(events_df, home_team, 'Block'),
        "blocks_away": _num_event_type(events_df, away_team, 'Block'),   
        "duels_won_home": _num_duels_won(events_df, home_team),
        "duels_won_away": _num_duels_won(events_df, away_team), 
        "tackles_home": _num_tackles(events_df, home_team),
        "tackles_away": _num_tackles(events_df, away_team),
        "tackles_success_ratio_home": _ratio_success_tackles(events_df, home_team),
        "tackles_success_ratio_away": _ratio_success_tackles(events_df, away_team),
        "fouls_committed_home": _num_event_type(events_df, home_team, 'Foul Committed'),
        "fouls_committed_away": _num_event_type(events_df, away_team, 'Foul Committed'),
        "50_50_won_home": _num_50_50_won(events_df, home_team),
        "50_50_won_away": _num_50_50_won(events_df, away_team),
        "clearances_home": _num_event_type(events_df, home_team, 'Clearance'),
        "clearances_away": _num_event_type(events_df, away_team, 'Clearance'),
        "penaltys_committed_home": _num_penaltys_committed(events_df, home_team),
        "penaltys_committed_away": _num_penaltys_committed(events_df, away_team),
        "key_errors_home": _num_event_type(events_df, home_team, 'Error'),
        "key_errors_away": _num_event_type(events_df, away_team, 'Error'),
        "miscontrols_home": _num_event_type(events_df, home_team, 'Miscontrol'),
        "miscontrols_away": _num_event_type(events_df, away_team, 'Miscontrol'),
        "yellow_cards_home": _num_cards_color_selected(events_df, home_team, "Yellow"),
        "yellow_cards_away": _num_cards_color_selected(events_df, away_team, "Yellow"),
        "red_cards_home": _num_cards_color_selected(events_df, home_team, "Red"),
        "red_cards_away": _num_cards_color_selected(events_df, away_team, "Red"),
        ## presión
        "pressures_home": _num_event_type(events_df, home_team, 'Pressure'),
        "pressures_away": _num_event_type(events_df, away_team, 'Pressure'),
        "counterpress_home": _num_counterpress(events_df, home_team),
        "counterpress_away": _num_counterpress(events_df, away_team),
        "pressures_in_attacking_third_home": _num_pressures_in_attacking_third(events_df, home_team),
        "pressures_in_attacking_third_away": _num_pressures_in_attacking_third(events_df, away_team),
        ## otros
        "offsides_home": _num_offsides(events_df, home_team),
        "offsides_away": _num_offsides(events_df, away_team),
        "dribbles_home": _num_event_type(events_df, home_team, 'Dribble'),
        "dribbles_away": _num_event_type(events_df, away_team, 'Dribble'),
        "dribbles_success_ratio_home": _ratio_success_dribbles(events_df, home_team),
        "dribbles_success_ratio_away": _ratio_success_dribbles(events_df, away_team),
        "injury_substitutions_home": _num_substitutions_because_of_injury(events_df, home_team),
        "injury_substitutions_away": _num_substitutions_because_of_injury(events_df, away_team),
        "players_off_home": _num_players_off(events_df, home_team),
        "players_off_away": _num_players_off(events_df, away_team),
        "dispossessed_home": _num_event_type(events_df, home_team, 'Dispossessed'),
        "dispossessed_away": _num_event_type(events_df, away_team, 'Dispossessed'),
        "counterattacks_home": _num_counterattacks(events_df, home_team),
        "counterattacks_away": _num_counterattacks(events_df, away_team),
        "possession_percentage_home": possession_percentage_home,
        "possession_percentage_away": possession_percentage_away,
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


def _num_interceptions_won(events_df, team):
    '''
    Calculate the number of interceptions won for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        int: The number of interceptions won.
    '''
    num_interceptions_won = 0
    if 'interception_outcome' in events_df.columns:
        num_interceptions_won = events_df[(events_df['team'] == team) & (events_df['type'] == 'Interception') & 
                                          (events_df['interception_outcome'].isin(["Success","Success In Play","Success Out","Won"]))].shape[0]
    return num_interceptions_won


def _num_recoveries(events_df, team):
    '''
    Calculate the number of recoveries for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        int: The number of recoveries.
    '''
    num_recoveries = 0
    if 'ball_recovery_offensive' in events_df.columns:
        num_recoveries = events_df[(events_df['team'] == team) & (events_df['type'] == 'Ball Recovery') & 
                                   (events_df['ball_recovery_offensive'] == True)].shape[0]
    return num_recoveries


def _num_duels_won(events_df, team):
    '''
    Calculate the number of duels won for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        int: The number of duels won.
    '''
    num_duels_won = 0
    if 'duel_outcome' in events_df.columns:
        num_duels_won = events_df[(events_df['team'] == team) & (events_df['type'] == 'Duel') & 
                                  (events_df['duel_outcome'].isin(["Won","Success","Success In Play","Success Out"]))].shape[0]
    return num_duels_won


def _num_tackles(events_df, team):
    '''
    Calculate the number of tackles for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        int: The number of tackles.
    '''
    num_tackles = 0
    if 'duel_type' in events_df.columns:
        num_tackles += events_df[(events_df['team'] == team) & (events_df['type'] == 'Duel') & 
                                (events_df['duel_type'] == "Tackle")].shape[0]
    if 'goalkeeper_type' in events_df.columns:
        num_tackles += events_df[(events_df['team'] == team) & (events_df['type'] == 'Goal Keeper') & 
                                (events_df['goalkeeper_type'] == "Smother")].shape[0]
    return num_tackles


def _ratio_success_tackles(events_df, team):
    '''
    Calculate the ratio of successful tackles for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        float: The ratio of successful tackles.
    '''
    total_tackles = _num_tackles(events_df, team)
    successful_tackles = 0
    if 'duel_type' in events_df.columns and 'duel_outcome' in events_df.columns:
        successful_tackles += events_df[(events_df['team'] == team) & (events_df['type'] == 'Duel') &
                                        (events_df['duel_type'] == "Tackle") &
                                        (events_df['duel_outcome'].isin(["Won","Success","Success In Play","Success Out"]))].shape[0]
    if 'goalkeeper_type' in events_df.columns and 'goalkeeper_outcome' in events_df.columns:
        successful_tackles += events_df[(events_df['team'] == team) & (events_df['type'] == 'Goal Keeper') & 
                                (events_df['goalkeeper_type'] == "Smother") &
                                (events_df['goalkeeper_outcome'].isin(["Won","Success","Success In Play","Success Out"]))].shape[0]
    return successful_tackles / total_tackles if total_tackles > 0 else 0.0


def _num_50_50_won(events_df, team):
    '''
    Calculate the number of 50/50 won for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        int: The number of 50/50 won.
    '''
    num_50_50_won = 0
    if '50_50' in events_df.columns:
        num_50_50_won = events_df[(events_df['team'] == team) & (events_df['type'] == '50/50') & 
                                  (events_df['50_50'].apply(
                                      lambda x: isinstance(x, dict) and x.get('outcome') and x['outcome'].get('name') in ["Won", "Success To Team"]
                                    ))].shape[0]
    return num_50_50_won


def _num_penaltys_committed(events_df, team):
    '''
    Calculate the number of penaltys committed for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        int: The number of penaltys committed.
    '''
    _num_penaltys_committed = 0
    if 'foul_committed_penalty' in events_df.columns:
        _num_penaltys_committed = events_df[(events_df['team'] == team) & (events_df['type'] == 'Foul Committed') & 
                                           (events_df['foul_committed_penalty'] == True)].shape[0]
    return _num_penaltys_committed


def _num_cards_color_selected(events_df, team, card_color):
    '''
    Calculate the number of selected color cards for a specific team. Since the second yellow card 
    implies a red card, we will count it as a red card.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
        card_color (str): The color of the card.
    returns:
        int: The number of selected color cards.
    '''
    num_cards_color = 0
    if 'foul_committed_card' in events_df.columns:
        if card_color == "Yellow":
            num_cards_color += events_df[(events_df['team'] == team) & (events_df['type'] == 'Foul Committed') & 
                                        (events_df['foul_committed_card'] == "Yellow Card")].shape[0]
        elif card_color == "Red":
            num_cards_color += events_df[(events_df['team'] == team) & (events_df['type'] == 'Foul Committed') & 
                                        (events_df['foul_committed_card'].isin(["Red Card","Second Yellow"]))].shape[0]
    if 'bad_behaviour_card' in events_df.columns:
        if card_color == "Yellow":
            num_cards_color += events_df[(events_df['team'] == team) & (events_df['type'] == 'Bad Behaviour') & 
                                        (events_df['bad_behaviour_card'] == "Yellow Card")].shape[0]
        elif card_color == "Red":
            num_cards_color += events_df[(events_df['team'] == team) & (events_df['type'] == 'Bad Behaviour') & 
                                        (events_df['bad_behaviour_card'].isin(["Red Card","Second Yellow"]))].shape[0]
    return num_cards_color


def _num_counterpress(events_df, team):
    '''
    Calculate the number of counterpress for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        int: The number of counterpress.
    '''
    num_counterpress = 0
    if 'counterpress' in events_df.columns:
        num_counterpress = events_df[(events_df['team'] == team) & (events_df['type'] == 'Pressure') & 
                                     (events_df['counterpress'] == True)].shape[0]
    return num_counterpress


def _num_pressures_in_attacking_third(events_df, team):
    '''
    Calculate the number of pressures in the attacking third for a specific team.
    Attacking third is defined as the area from 80 to 120 in x-axis and from 0 to 80 in y-axis.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        int: The number of pressures in the attacking third.
    '''
    return events_df[(events_df['team'] == team) & (events_df['type'] == 'Pressure') &
                     (events_df['location'].notnull()) &
                     (events_df['location'].apply(lambda loc: isinstance(loc, list) and 80 <= loc[0] <= 120 and 0 <= loc[1] <= 80))].shape[0]


def _num_offsides(events_df, team):
    '''
    Calculate the number of offsides for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        int: The number of offsides.
    '''
    non_pass_offsides = events_df[(events_df['team'] == team) & (events_df['type'] == 'Offside')].shape[0]
    pass_offsides = events_df[(events_df['team'] == team) & (events_df['type'] == 'Pass') &
                              (events_df['pass_outcome'] == "Pass Offside")].shape[0]
    return non_pass_offsides + pass_offsides


def _ratio_success_dribbles(events_df, team):
    '''
    Calculate the ratio of successful dribbles for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        float: The ratio of successful dribbles.
    '''
    total_dribbles = _num_event_type(events_df, team, 'Dribble')
    successful_dribbles = 0
    if 'dribble_outcome' in events_df.columns:
        successful_dribbles = events_df[(events_df['team'] == team) & (events_df['type'] == 'Dribble') & 
                                        (events_df['dribble_outcome'] == "Complete")].shape[0]
    return successful_dribbles / total_dribbles if total_dribbles > 0 else 0.0


def _num_substitutions_because_of_injury(events_df, team):
    '''
    Calculate the number of substitutions because of injury for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        int: The number of substitutions because of injuries.
    '''
    num_substitutions_because_of_injuries = 0
    if 'substitution_outcome' in events_df.columns:
        num_substitutions_because_of_injuries = events_df[(events_df['team'] == team) & (events_df['type'] == 'Substitution') & 
                                                          (events_df['substitution_outcome'] == "Injury")].shape[0]
    return num_substitutions_because_of_injuries


def _num_players_off(events_df, team):
    '''
    Calculate the number of players off (injured players who have to leave the field 
    without making a substitution because there is no one left) for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        int: The number of players off.
    '''
    num_players_off = 0
    if 'permanent' in events_df.columns:
        num_players_off = events_df[(events_df['team'] == team) & (events_df['type'] == 'Player Off') & 
                                    (events_df['permanent'] == True)].shape[0]
    return num_players_off


def _num_counterattacks(events_df, team):
    '''
    Calculate the number of counterattacks for a specific team.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        team (str): The team.
    returns:
        int: The number of counterattacks.
    '''
    return events_df[(events_df['team'] == team) & 
                     (events_df['play_pattern'] == "From Counter")]['possession'].nunique()


def _possession_percentage(events_df, home_team, away_team):
    '''
    Calculate the possession percentage for a specific team. Ball possession refers to 
    the amount of time a team has control of the ball during a match.
    params:
        events_df (DataFrame): A DataFrame containing the events.
        home_team (str): Home team.
        away_team (str): Away team.
    returns:
        float: The possession percentage.
    '''
    possessions_percentage = _calculate_possession_percentage(events_df.copy())
    home_team_possession_percentage = possessions_percentage.get(home_team, 0.0)
    away_team_possession_percentage = possessions_percentage.get(away_team, 0.0)
    return home_team_possession_percentage, away_team_possession_percentage


def _calculate_possession_percentage(events_df):
    """
    Calculate the possession percentage for each team in a specific team.
    params:
        events_df (DataFrame): DataFrame con los eventos del partido.
    returns:
        dict: Porcentajes de posesión para cada equipo.
    """
    events_df = events_df.sort_values(by='timestamp')
    # identificamos al equipo de cada posesión
    possession_durations = events_df.groupby('possession').apply(
        lambda x: {
            'team': x['possession_team'].iloc[0],
            'duration': (pd.to_datetime(x['timestamp'].iloc[-1]) - pd.to_datetime(x['timestamp'].iloc[0])).total_seconds()
        }
    )
    # creamos un dataframe con la duración de cada posesión
    possession_df = pd.DataFrame(list(possession_durations), columns=['team', 'duration'])
    # sumamos el tiempo de posesión de cada equipo
    possession_time_by_team = possession_df.groupby('team')['duration'].sum()
    total_time = possession_time_by_team.sum()
    # calculamos el porcentaje de posesión para cada equipo
    possession_percentage = (possession_time_by_team / total_time)
    return possession_percentage.to_dict()

