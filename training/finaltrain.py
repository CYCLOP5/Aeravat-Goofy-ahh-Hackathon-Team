from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

index_names = ['unit_number', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['sensor_{}'.format(i) for i in range(1,22)]
col_names = index_names + setting_names + sensor_names

train_files = ['data/train_FD001.txt', 'data/train_FD002.txt', 'data/train_FD003.txt', 'data/train_FD004.txt']
test_files = ['data/test_FD001.txt', 'data/test_FD002.txt', 'data/test_FD003.txt', 'data/test_FD004.txt']
rul_files = ['data/RUL_FD001.txt', 'data/RUL_FD002.txt', 'data/RUL_FD003.txt', 'data/RUL_FD004.txt']

train_dfs = []
test_dfs = []
rul_dfs = []

for train_file, test_file, rul_file in zip(train_files, test_files, rul_files):
    train_dfs.append(pd.read_csv(train_file, sep='\s+', header=None, index_col=False, names=col_names))
    test_dfs.append(pd.read_csv(test_file, sep='\s+', header=None, index_col=False, names=col_names))
    rul_dfs.append(pd.read_csv(rul_file, sep='\s+', header=None, index_col=False, names=['RUL']))

df_train = pd.concat(train_dfs)
df_test = pd.concat(test_dfs)
y_test = pd.concat(rul_dfs)

def add_RUL_column(df):
    max_time_cycles = df.groupby(by='unit_number')['time_cycles'].max()
    merged = df.merge(max_time_cycles.to_frame(name='max_time_cycle'), left_on='unit_number', right_index=True)
    merged["RUL"] = merged["max_time_cycle"] - merged['time_cycles']
    merged = merged.drop("max_time_cycle", axis=1)
    return merged

train = add_RUL_column(df_train)
test = add_RUL_column(df_test)

drop_labels = ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3', 'RUL']
X_train = train.drop(drop_labels, axis=1)
y_train = train['RUL']
X_test = test.drop(drop_labels, axis=1)
y_test = y_test.values.flatten()

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train_scaled, y_train, test_size=0.4, random_state=42)

svr_model = SVR()
svr_model.fit(X_train_split, y_train_split)
y_pred_train = svr_model.predict(X_train_split)
y_pred_test = svr_model.predict(X_test_scaled).flatten()

strtrain_rmse = mean_squared_error(y_train_split, y_pred_train, squared=False)
train_r2 = r2_score(y_train_split, y_pred_train)
train_mae = mean_absolute_error(y_train_split, y_pred_train)

test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)

print("Train set metrics:")
print(f"RMSE: {train_rmse}")
print(f"R^2: {train_r2}")
print(f"MAE: {train_mae}")
print("\nTest set metrics:")
print(f"RMSE: {test_rmse}")
print(f"R^2: {test_r2}")
print(f"MAE: {test_mae}")
