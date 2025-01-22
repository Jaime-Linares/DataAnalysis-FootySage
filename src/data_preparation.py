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


def reduce_dimensionality(matches_processed_df):
    '''
    Reduces the dimensionality of the data.
    params:
        matches_processed_df (DataFrame): A DataFrame containing the processed data.
    returns:
        DataFrame: A DataFrame containing the reduced data.
    '''
    matches_processed_reduced_df = _percentage_of_shots_home_team(matches_processed_df)
    matches_processed_df = _percentage_of_shots_high_xG_home_team(matches_processed_reduced_df)
    matches_processed_df = _percentage_of_shots_inside_area_home_team(matches_processed_reduced_df)
    return matches_processed_reduced_df


def _percentage_of_shots_home_team(matches_processed_df):
    '''
    Calculates the percentage of shots of the home team from all the match.
    If there have been no shots in the match 0.5 for each team is assigned.
    params:
        matches_processed_df (DataFrame): A DataFrame containing the processed data.
    returns:
        DataFrame: A DataFrame containing the percentage of shots of the home team.
    '''
    total_shots = matches_processed_df['total_shots_home'] + matches_processed_df['total_shots_away']
    matches_processed_df['percentage_shots_home'] = (matches_processed_df['total_shots_home'] / total_shots).fillna(0.5)
    matches_processed_df.drop(['total_shots_home', 'total_shots_away'], axis=1, inplace=True)
    return matches_processed_df


def _percentage_of_shots_high_xG_home_team(matches_processed_df):
    '''
    Calculates the percentage of shots with high xG of the home team from all the match.
    If there have been no shots with high xG in the match 0.5 for each team is assigned.
    params:
        matches_processed_df (DataFrame): A DataFrame containing the processed data.
    returns:
        DataFrame: A DataFrame containing the percentage of shots with high xG of the home team.
    '''
    total_shots_high_xG = matches_processed_df['shots_high_xG_home'] + matches_processed_df['shots_high_xG_away']
    matches_processed_df['percentage_shots_high_xG_home'] = (matches_processed_df['shots_high_xG_home'] / total_shots_high_xG).fillna(0.5)
    matches_processed_df.drop(['shots_high_xG_home', 'shots_high_xG_away'], axis=1, inplace=True)
    return matches_processed_df


def _percentage_of_shots_inside_area_home_team(matches_processed_df):
    '''
    Calculates the percentage of shots inside the area of the home team from all the match.
    If there have been no shots inside the area in the match 0.5 for each team is assigned.
    params:
        matches_processed_df (DataFrame): A DataFrame containing the processed data.
    returns:
        DataFrame: A DataFrame containing the percentage of shots inside the area of the home team.
    '''
    total_shots_inside_area = matches_processed_df['shots_inside_area_home'] + matches_processed_df['shots_inside_area_away']
    matches_processed_df['percentage_shots_inside_area_home'] = (matches_processed_df['shots_inside_area_home'] / total_shots_inside_area).fillna(0.5)
    matches_processed_df.drop(['shots_inside_area_home', 'shots_inside_area_away'], axis=1, inplace=True)
    return matches_processed_df

