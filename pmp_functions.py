'''
Module containing the functions used in predicting molecular properties competition
'''
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Helper Functions
def replace_columns_type(columns, new_cols):
    for i in range(0,len(new_cols)):
        columns.pop(-1)
    columns.extend(new_cols)
    return columns

def replace_columns_atom_0(columns, new_cols):
    for i in range(0,len(new_cols)):
        columns.pop(-1)
    for i in range(0, len(new_cols)):
        new_cols[i] = new_cols[i] + '_atom_0'
    columns.extend(new_cols)
    return columns

def replace_columns_atom_1(columns, new_cols):
    for i in range(0, len(new_cols)):
        columns.pop(-1)
    for i in range(0, len(new_cols)):
        new_cols[i] = new_cols[i] + '_atom_1'
    columns.extend(new_cols)
    return columns

# Main Functions
def merge_structures(df, structures):
    df = pd.merge(df, structures, how='inner', 
                  left_on=['molecule_name','atom_index_0'], 
                  right_on=['molecule_name','atom_index']
                  )
    df.rename(columns={'atom':'atom_0','x':'x0','y':'y0','z':'z0'}, 
              inplace=True
              )
    df.drop('atom_index', axis=1, inplace=True)
    df = pd.merge(df, structures, how='left', 
                  left_on=['molecule_name','atom_index_1'], 
                  right_on=['molecule_name','atom_index']
                  )
    df.drop('atom_index', axis=1, inplace=True)
    df.rename(columns={'atom':'atom_1','x':'x1','y':'y1','z':'z1'}, 
              inplace=True
              )
    return df

def categorical_encode(df, one_hot):
    type_ = df[['type']]
    atom_0 = df[['atom_0']]
    atom_1 = df[['atom_1']]
    
    one_hot.fit(type_)
    type_cat = pd.DataFrame(one_hot.transform(type_).toarray())
    df = pd.concat([df, type_cat],axis=1)
    df.columns = replace_columns_type(df.columns.tolist(), 
                                      one_hot.categories_[0].tolist()
                                      )
    
    one_hot.fit(atom_0)
    atom_0_cat = pd.DataFrame(one_hot.transform(atom_0).toarray())
    df = pd.concat([df, atom_0_cat], axis=1)
    df.columns = replace_columns_atom_0(df.columns.tolist(), 
                                        one_hot.categories_[0].tolist()
                                        )
    
    one_hot.fit(atom_1)
    atom_1_cat = pd.DataFrame(one_hot.transform(atom_1).toarray())
    df = pd.concat([df, atom_1_cat], axis=1)
    df.columns = replace_columns_atom_1(df.columns.tolist(), 
                                        one_hot.categories_[0].tolist()
                                        )
    return df

# Trig Transfroms
def trig_transforms(df):
    # Euclidean Distance between atom coordinates and atoms
    df['x_distance'] = np.sqrt(np.square((df['x0'].to_numpy() - df['x1'].to_numpy())))
    df['y_distance'] = np.sqrt(np.square((df['y0'].to_numpy() - df['y1'].to_numpy())))
    df['z_distance'] = np.sqrt(np.square(df['z0'].to_numpy() - df['z1'].to_numpy()))  
    x_component = np.square(df['x_distance'].to_numpy())
    y_component = np.square(df['y_distance'].to_numpy())
    z_component = np.square(df['z_distance'].to_numpy())
    df['distance'] = np.sqrt(x_component + y_component + z_component)
    '''
    distance = lambda row: ((row.x0-row.x1)**2 + (row.y0-row.y1)**2 + (row.z0-row.z1)**2)**0.5
    df['distance'] = df.apply(distance, axis=1)
    '''
    # 3D Angles between atoms
    df['alpha_angle'] = np.arccos((df['x_distance'] / df['distance']).to_numpy())
    df['beta_angle'] = np.arccos((df['y_distance'] / df['distance']).to_numpy())
    df['gamma_angle'] = np.arccos((df['z_distance'] / df['distance']).to_numpy())
    '''
    # Angle Trig Fuctions
    df['alpha_cos'] = np.cos(df['alpha_angle'].to_numpy())
    df['beta_cos'] = np.cos(df['beta_angle'].to_numpy())
    df['gamma_cos'] = np.cos(df['gamma_angle'].to_numpy())
    df['alpha_sin'] = np.sin(df['alpha_angle'].to_numpy())
    df['beta_sin'] = np.sin(df['beta_angle'].to_numpy())
    df['gamma_sin'] = np.sin(df['gamma_angle'].to_numpy())
    df['alpha_tan'] = np.tan(df['alpha_angle'].to_numpy())
    df['beta_tan'] = np.tan(df['beta_angle'].to_numpy())
    df['gamma_tan'] = np.tan(df['gamma_angle'].to_numpy())
    # Angle Hyperbolic Trig Functions
    df['alpha_cosh'] = np.cosh(df['alpha_angle'].to_numpy())
    df['beta_cosh'] = np.cosh(df['beta_angle'].to_numpy())
    df['gamma_cosh'] = np.cosh(df['gamma_angle'].to_numpy())
    df['alpha_sinh'] = np.sinh(df['alpha_angle'].to_numpy())
    df['beta_sinh'] = np.sinh(df['beta_angle'].to_numpy())
    df['gamma_sinh'] = np.sinh(df['gamma_angle'].to_numpy())
    df['alpha_tanh'] = np.tanh(df['alpha_angle'].to_numpy())
    df['beta_tanh'] = np.tanh(df['beta_angle'].to_numpy())
    df['gamma_tanh'] = np.tanh(df['gamma_angle'].to_numpy())
    # Means by Molecular Structure
    means = df.groupby(['type']).mean()
    print(means.columns)
    means.drop(columns=['id', 'atom_index_0', 'atom_index_1', 
                        'x0', 'y0', 'z0', 'x1', 'y1',
                        'z1', '1JHC', '1JHN', '2JHC', 
                        '2JHH', '2JHN', '3JHC', '3JHH', 
                        '3JHN','H_atom_0', 'C_atom_1', 'H_atom_1', 
                        'N_atom_1'], inplace=True)  
    # STD by Molecular Structure
    std = df.groupby(['type']).std()
    std.drop(columns=['id', 'atom_index_0', 'atom_index_1', 
                        'x0', 'y0', 'z0', 'x1', 'y1',
                        'z1', '1JHC', '1JHN', '2JHC', 
                        '2JHH', '2JHN', '3JHC', '3JHH', 
                        '3JHN','H_atom_0', 'C_atom_1', 'H_atom_1',
                        'N_atom_1'], inplace=True)  
    df = df.join(means, on='type', how='inner', rsuffix='_mean')
    df = df.join(std, on='type', how='inner', rsuffix='_std') 
    '''
    # Angle Inverse Functions
    '''
    We will ignore the inverse functions for now, 
    until we decide what to do with the 1/0 problem
    
    df['alpha_sec'] = 1 / df['alpha_cos']
    df['beta_sec'] = 1 / df['beta_cos']
    df['gamma_sec'] = 1 / df['gamma_cos']
    df['alpha_csc'] = 1 / df['alpha_sin']
    df['beta_csc']  = 1 / df['beta_sin']
    df['gamma_csc'] = 1 / df['gamma_sin']
    df['alpha_cot'] = 1 / df['alpha_tan']
    df['beta_cot'] = 1 / df['beta_tan']
    df['gamma_cot'] = 1 / df['gamma_tan']
    # Angle Hyperbolic Inverse Functions
    df['alpha_sech']  = 1 / df['alpha_cosh']
    df['beta_sech'] = 1 / df['beta_cosh']
    df['gamma_sech'] = 1 / df['gamma_cosh']
    df['alpha_csch'] = 1 / df['alpha_sinh']
    df['beta_csch'] = 1 / df['beta_sinh']
    df['gamma_csch'] = 1 / df['gamma_sinh']
    df['alpha_coth'] = 1 / df['alpha_tanh']
    df['beta_coth'] = 1 / df['beta_tanh']
    df['gamma_coth'] = 1 / df['gamma_tanh']
    '''
    return df
    
def drop_columns(df):
    df.drop(columns=['molecule_name'], inplace=True)
    df.drop(columns=['x0','y0','z0','x1','y1','z1'], inplace=True)
    df.drop(columns=["H_atom_0"], inplace=True)
    df.drop(columns=['type','atom_1','atom_0'], inplace=True)
    return df
    
def create_cat_and_id_sets(df, option=0):
    df_cat = df[['1JHC','1JHN','2JHC','2JHH',
                 '2JHN','3JHC','3JHH','3JHN',
                 'C_atom_1','H_atom_1','N_atom_1']]
    df.drop(columns=['atom_index_0','atom_index_1',
                     '1JHC','1JHN','2JHC','2JHH',
                     '2JHN','2JHC','3JHC','3JHH',
                     '3JHN','C_atom_1','H_atom_1',
                     'N_atom_1'], inplace=True)
    df_id = df['id']
    df.drop(columns=['id'], inplace=True)
    if option == 0:
        return df, df_cat, df_id
    else:
        return df, df_id

def create_target_set(df):
    df_target = df[['scalar_coupling_constant']]
    df.drop(columns=['scalar_coupling_constant'], inplace=True)
    return df, df_target

def STD_Scaler(df):
    columns_ = df.columns
    scaler = StandardScaler()
    # Test...
    '''
    for column in df.columns:
        print('Trying . . .', column)
        scaler.fit(df[[column]])
        scaler.transform(df[[column]])
    '''
    scaler.fit(df)
    df = scaler.transform(df)
    df = pd.DataFrame(df)
    df.columns = columns_
    return df

def Min_Max_Scaler(df):
    columns_ = df.columns
    min_max = MinMaxScaler()
    min_max.fit(df)
    df = min_max.transform(df)
    df = pd.DataFrame(df)
    df.columns = columns_
    return df

def join_cat(df, df_cat):
    df = df.join(df_cat)
    return df

def drops(df):
    #df.drop(columns=['x_distance','y_distance','z_distance',
     #                '2JHN','3JHC','3JHH','3JHN'], inplace=True)
    return df

def create_X_y_train_val(df, df_target):
    X = df.to_numpy()
    y = df_target
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def create_X_test(df):
    X_test = df.to_numpy()
    return X_test

def read_csv():
    
    DATA_PATH = "C:/Users/edher/Desktop/ML/datasets/champs-scalar-coupling"
    
    dipole_moments_file_f = "dipole_moments.csv"
    magnetic_shielding_tensors_f = "magnetic_shielding_tensors.csv"
    mulliken_charges_f = "mulliken_charges.csv"
    potential_energy_f = "potential_energy.csv"
    scalar_coupling_contributions_f = "scalar_coupling_contribution.csv"
    
    structures_f = "structures.csv"
    test_f = "test.csv"
    train_f = "train.csv"    
    
    train_set = pd.read_csv(os.path.join(DATA_PATH, train_f))
    test_set = pd.read_csv(os.path.join(DATA_PATH, test_f))
    structures = pd.read_csv(os.path.join(DATA_PATH, structures_f))  
    return train_set, test_set, structures

def processing_pipeline(option, norm='none'):
    
    train_set, test_set, structures = read_csv()
    
    one_hot = OneHotEncoder()
    
    if (option == 0) or (option == 1):
        #train_set = train_set.head(1000)
        train_set, train_target = create_target_set(train_set)
        train_set = merge_structures(train_set, structures)
        train_set = categorical_encode(train_set, one_hot)
        train_set = trig_transforms(train_set)
        train_set = drop_columns(train_set)
        train_set, train_cat, train_id = create_cat_and_id_sets(train_set, option=0)
        if norm == 'std_scaler':
            train_set = STD_Scaler(train_set)
        if norm == 'min_max':
            train_set = Min_Max_Scaler(train_set)
        train_set = join_cat(train_set, train_cat)
        train_set = drops(train_set)
        X_train, X_val, y_train, y_val = create_X_y_train_val(train_set, train_target)
    
    if (option == 0) or (option == 2):
        #test_set = test_set.head(500)
        test_set = merge_structures(test_set, structures)
        test_set = categorical_encode(test_set, one_hot)
        test_set = trig_transforms(test_set)
        test_set = drop_columns(test_set)
        test_set, test_cat, test_id = create_cat_and_id_sets(test_set, option=0)
        if norm == 'std_scaler':
            test_set = STD_Scaler(test_set)
        if norm == 'min_max':
            test_set = Min_Max_Scaler(test_set)
        test_set = join_cat(test_set, test_cat)
        test_set = drops(test_set)
        X_test = create_X_test(test_set)
    
    if (option == 2):
        return test_set, X_test, test_id
    elif (option == 1):
        return train_set, X_train, X_val, y_train, y_val, train_target
    else:
        return train_set, test_set, X_train, X_val, y_train, y_val, train_target, X_test, test_id

def gradient_boosting(model, x_train, y_train, boosts):
    models = [model]
    for i in range(0, boosts + 1):
        models.append(model)
    # Initialize
    models[0].fit(x_train, y_train)
    y_predictions = models[0].predict(x_train)
    y = y_train
    # BOOST
    for i in range(1, boosts + 1):
        residuals = y - y_predictions
        models[i].fit(x_train, residuals)
        y_predictions = models[i].predict(x_train)
        y = residuals       
    return models

def boosted_predictions(models, X):
    predictions = models[0].predict(X)
    for i in range(1, len(models)):
        predictions += models[i].predict(predictions)
    predictions = pd.DataFrame(data={'id': ids, 'scalar_coupling_constant': predictions})
    return predictions
    
def evaluation_metrics(model, predictions, train, actual):
    mse = mean_squared_error(predictions, actual)
    rmse = np.sqrt(mse)
    R2 = model.score(train, actual)
    print("Mean Square Error:", mse)
    print("Root Mean Square Error", rmse)
    print("R2 Score:", R2)
    return mse, rmse, R2

def make_predictions(model, test_set, ids):
    predictions = model.predict(test_set)
    predictions = pd.DataFrame(data={'id': ids, 'scalar_coupling_constant': predictions})
    return predictions

def predictions_to_csv(predictions_df, name):
    predictions_df.to_csv(name + '.csv', index=False)
    
def unit_test():
    t1 = time.clock()
    train_set, test_set, X_train, X_val, y_train, y_val, train_target, X_test, test_id = processing_pipeline(0, norm='std_scaler')
    t2 = time.clock()
    print(test_set.info())
    print(test_set.head(10))
    print('Elapsed Time:', t2-t1)
    
#unit_test()
    
