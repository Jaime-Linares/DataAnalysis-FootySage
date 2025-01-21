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

