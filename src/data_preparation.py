# encoding:utf-8
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split



def code_categorical_data_multiclass(processed_data):
    '''
    Encodes the categorical data with more than 2 classes.
    params:
        processed_data (DataFrame): A DataFrame containing the proccesed_data.
    returns:
        tuple: A tuple containing the processed data and the encoder.
    '''
    encoder = LabelEncoder()
    processed_data = encoder.fit_transform(processed_data)
    return processed_data, encoder


def divide_data_in_train_test(data, target, test_size=0.2):
    '''
    Divides the data into training and test sets.
    params:
        data (DataFrame): A DataFrame containing the data.
        target (DataFrame): A DataFrame containing the target.
        test_size (float): The size of the test set.
    returns:
        tuple: A tuple containing the training and test sets.
    '''
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=42, stratify=target)
    return X_train, X_test, y_train, y_test


def scale_data_train_test(X_train, X_test, scaler):
    '''
    Scales the data.
    params:
        X_train (DataFrame): A DataFrame containing the training data.
        X_test (DataFrame): A DataFrame containing the test data.
        scaler (str): The scaler to use.
    returns:
        tuple: A tuple containing the scaled training and test data.
    '''
    if scaler == 'StandardScaler':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif scaler == 'MinMaxScaler':
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        raise ValueError('Invalid scaler. Valid scalers are: StandardScaler, MinMaxScaler')
    return X_train, X_test


# --- FUNCIÓN PRINCIPAL PARA REDUCIR LA DIMENSIONALIDAD ----------------------------------------------------------------------------------------------------------
def reduce_dimensionality(matches_processed_df):
    '''
    Reduces the dimensionality of the data.
    params:
        matches_processed_df (DataFrame): A DataFrame containing the processed data.
    returns:
        DataFrame: A DataFrame containing the reduced data.
    '''
    # estadísticas generales del partido
    ## tiros
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_df, 'total_shots_home', 'total_shots_away', 'percentage_total_shots_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'shots_high_xG_home', 'shots_high_xG_away', 'percentage_shots_high_xG_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'shots_inside_area_home', 'shots_inside_area_away', 'percentage_shots_inside_area_home')
    matches_processed_reduced_df = _percentage_of_shots_with_body_part_home_team(matches_processed_reduced_df, 'foot')
    matches_processed_reduced_df = _percentage_of_shots_with_body_part_home_team(matches_processed_reduced_df, 'head')
    matches_processed_reduced_df = _percentage_of_shots_with_body_part_home_team(matches_processed_reduced_df, 'other')
    ## pases
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'total_passes_home', 'total_passes_away', 'percentage_total_passes_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'key_passes_home', 'key_passes_away', 'percentage_key_passes_home')
    matches_processed_reduced_df = _difference_of_passes_needed_to_make_a_shot_home_team(matches_processed_reduced_df)
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'crosses_home', 'crosses_away', 'percentage_crosses_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'corners_home', 'corners_away', 'percentage_corners_home')
    ## defensa
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'interceptions_won_home', 'interceptions_won_away', 'percentage_interceptions_won_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'recoveries_home', 'recoveries_away', 'percentage_recoveries_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'blocks_home', 'blocks_away', 'percentage_blocks_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'duels_won_home', 'duels_won_away', 'percentage_duels_won_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'tackles_home', 'tackles_away', 'percentage_tackles_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'fouls_committed_home', 'fouls_committed_away', 'percentage_fouls_committed_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, '50_50_won_home', '50_50_won_away', 'percentage_50_50_won_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'clearances_home', 'clearances_away', 'percentage_clearance_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'penaltys_committed_home', 'penaltys_committed_away', 'percentage_penaltys_committed_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'key_errors_home', 'key_errors_away', 'percentage_key_errors_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'miscontrols_home', 'miscontrols_away', 'percentage_miscontrols_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'yellow_cards_home', 'yellow_cards_away', 'percentage_yellow_cards_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'red_cards_home', 'red_cards_away', 'percentage_red_cards_home')
    ## presión
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'pressures_home', 'pressures_away', 'percentage_pressures_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'counterpress_home', 'counterpress_away', 'percentage_counterpress_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'pressures_in_attacking_third_home', 'pressures_in_attacking_third_away', 'percentage_pressures_in_attacking_third_home')
    ## otros
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'offsides_home', 'offsides_away', 'percentage_offsides_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'dribbles_home', 'dribbles_away', 'percentage_dribbles_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'injury_substitutions_home', 'injury_substitutions_away', 'percentage_injury_substitutions_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'players_off_home', 'players_off_away', 'percentage_players_off_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'dispossessed_home', 'dispossessed_away', 'percentage_dispossessed_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'counterattacks_home', 'counterattacks_away', 'percentage_counterattacks_home')
    # estadística contextuales
    ## recuperaciones
    matches_processed_reduced_df = _percentage_of_recoveries_in_selected_third_home_team(matches_processed_reduced_df, 'attacking')
    matches_processed_reduced_df = _percentage_of_recoveries_in_selected_third_home_team(matches_processed_reduced_df, 'middle')
    matches_processed_reduced_df = _percentage_of_recoveries_in_selected_third_home_team(matches_processed_reduced_df, 'defensive')
    ## eventos bajo presión
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'shots_under_pressure_home', 'shots_under_pressure_away', 'percentage_shots_under_pressure_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'shots_inside_area_under_pressure_home', 'shots_inside_area_under_pressure_away', 'percentage_shots_inside_area_under_pressure_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'passes_under_pressure_home', 'passes_under_pressure_away', 'percentage_passes_under_pressure_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'passes_inside_area_under_pressure_home', 'passes_inside_area_under_pressure_away', 'percentage_passes_inside_area_under_pressure_home')
    ## jugadas a balón parado
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'set_piece_shots_home', 'set_piece_shots_away', 'percentage_set_piece_shots_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'set_piece_shots_inside_area_home', 'set_piece_shots_inside_area_away', 'percentage_set_piece_shots_inside_area_home')
    # tácticas
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'substitutions_home', 'substitutions_away', 'percentage_substitutions_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'tactical_substitutions_home', 'tactical_substitutions_away', 'percentage_tactical_substitutions_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'tactical_changes_home', 'tactical_changes_away', 'percentage_tactical_changes_home')
    matches_processed_reduced_df = _percentage_of_metric_home_team(matches_processed_reduced_df, 'formation_changes_home', 'formation_changes_away', 'percentage_formation_changes_home')

    return matches_processed_reduced_df


# --- FUNCIONES AUXILIARES PARA REDUCIR LA DIMENSIONALIDAD -------------------------------------------------------------------------------------------------------
def _percentage_of_metric_home_team(matches_processed_df, name_home_column, name_away_column, name_percentage_column):
    '''
    Calculates the percentage of a specified metric for the home team from the total in the match.
    If there have been no occurrences of the metric in the match, 0.5 is assigned to each team.
    params:
        matches_processed_df (DataFrame): A DataFrame containing the processed data.
        name_home_column (str): The column name for the home team's metric.
        name_away_column (str): The column name for the away team's metric.
        name_percentage_column (str): The column name for the resulting percentage.
    returns:
        DataFrame: A DataFrame containing the percentage of the specified metric for the home team.
    '''
    if name_home_column not in matches_processed_df.columns or name_away_column not in matches_processed_df.columns:
        raise ValueError(f'Invalid column names. Valid column names are: {matches_processed_df.columns}')
    total_metric = matches_processed_df[name_home_column] + matches_processed_df[name_away_column]
    matches_processed_df[name_percentage_column] = (matches_processed_df[name_home_column] / total_metric).fillna(0.5)
    matches_processed_df.drop([name_home_column, name_away_column], axis=1, inplace=True)
    return matches_processed_df


def _percentage_of_shots_with_body_part_home_team(matches_processed_df, body_part):
    '''
    Calculates the percentage of shots with a specific body part of the home team from all the match.
    If there have been no shots with the body part in the match 0.5 for each team is assigned.
    params:
        matches_df (DataFrame): A DataFrame containing the processed data.
        body_part (str): The body part to calculate the percentage.
    returns:
        DataFrame: A DataFrame containing the percentage of shots with the body part of the home team.
    '''
    if body_part not in ['foot', 'head', 'other']:
        raise ValueError('Invalid body part. Valid body parts are: foot, head, other')
    total_shots_body_part = matches_processed_df[f'shots_{body_part}_home'] + matches_processed_df[f'shots_{body_part}_away']
    matches_processed_df[f'percentage_shots_{body_part}_home'] = (matches_processed_df[f'shots_{body_part}_home'] / total_shots_body_part).fillna(0.5)
    matches_processed_df.drop([f'shots_{body_part}_home', f'shots_{body_part}_away'], axis=1, inplace=True)
    return matches_processed_df


def _difference_of_passes_needed_to_make_a_shot_home_team(matches_processed_df):
    '''
    Calculates the difference of passes needed to make a shot of the home team.
    params:
        matches_processed_df (DataFrame): A DataFrame containing the processed data.
    returns:
        DataFrame: A DataFrame containing the difference of passes needed to make a shot of the home team.
    '''
    matches_processed_df['difference_passes_needed_to_make_a_shot_home'] = matches_processed_df['passes_needed_to_make_a_shot_home'] - matches_processed_df['passes_needed_to_make_a_shot_away']
    matches_processed_df.drop(['passes_needed_to_make_a_shot_home', 'passes_needed_to_make_a_shot_away'], axis=1, inplace=True)
    return matches_processed_df


def _percentage_of_recoveries_in_selected_third_home_team(matches_processed_df, part):
    '''
    Calculates the percentage of recoveries in a selected third of the home team.
    params:
        matches_processed_df (DataFrame): A DataFrame containing the processed data.
        part (str): The selected third.
    returns:
        DataFrame: A DataFrame containing the percentage of recoveries in the selected third of the home team.
    '''
    if part not in ['defensive', 'middle', 'attacking']:
        raise ValueError('Invalid part. Valid parts are: defensive, middle, attacking')
    total_recoveries = matches_processed_df[f'recoveries_{part}_third_home'] + matches_processed_df[f'recoveries_{part}_third_away']
    matches_processed_df[f'percentage_recoveries_{part}_third_home'] = (matches_processed_df[f'recoveries_{part}_third_home'] / total_recoveries).fillna(0.5)
    matches_processed_df.drop([f'recoveries_{part}_third_home', f'recoveries_{part}_third_away'], axis=1, inplace=True)
    return matches_processed_df

