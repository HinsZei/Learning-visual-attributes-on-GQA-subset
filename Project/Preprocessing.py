import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE


# Remove the unused columns and code the colour columns
def training_data_color(path):
    path = 'data_train.csv'
    data = pd.read_csv(path)
    data.set_index('image', inplace=True)
    color = data['color']
    data.drop(columns=['x', 'y', 'w', 'h', 'id', 'texture', 'color'], inplace=True)
    data['color'] = color
    data = data.values
    X, y = data[:, :-1], data[:, -1]
    # encoding labels to 0-12
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    return X, y, encoder

# Remove the unused columns and code the texture columns
def training_data_texture(path):
    path = 'data_train.csv'
    data = pd.read_csv(path)
    data.set_index('image', inplace=True)
    texture = data['texture']
    data.drop(columns=['x', 'y', 'w', 'h', 'id', 'texture', 'color'], inplace=True)
    data['texture'] = texture
    data = data.values
    X, y = data[:, :-1], data[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    return X, y, encoder

# Read the test data remove the unused columns
def test_data(path):
    path = 'data_test.csv'
    data = pd.read_csv(path)
    data.set_index('image', inplace=True)
    data.drop(columns=['x', 'y', 'w', 'h', 'id'], inplace=True)
    data = data.values
    return data

# Fill in with median
def imputer():
    return SimpleImputer(strategy='median')


def scaler():
    return StandardScaler()


def sampling_SmoteTomek():
    # Due to the small number of samples in some classes (6),
    # I could only set the neighbouring points to 2 for validation
    smote = SMOTE(k_neighbors=2)
    sampler = SMOTETomek(random_state=0, smote=smote)
    return sampler

