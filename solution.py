import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Загрузка данных
print("Загрузка данных...")
df = pd.read_csv('train.zip', compression='zip', header=0, sep=',', quotechar='"')

# Преобразование строковых дат в datetime
print("Преобразование дат...")
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df = df.drop('dropoff_datetime', axis=1, errors='ignore')

# Сортировка данных по дате
df = df.sort_values(by='pickup_datetime')

# Разбиение на train и test
train_df, test_df = df[:10**6], df[10**6:]

# Добавление логарифма длительности поездки (если его нет)
if 'log_trip_duration' not in train_df.columns:
    train_df['log_trip_duration'] = np.log(train_df['trip_duration'])
    test_df['log_trip_duration'] = np.log(test_df['trip_duration'])

# Задание 1: График количества поездок в зависимости от дня недели
print("Построение графика по дням недели...")
train_df['day_of_week'] = train_df['pickup_datetime'].dt.dayofweek
day_counts = train_df['day_of_week'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
sns.barplot(x=day_counts.index, y=day_counts.values)
plt.title('Количество поездок в зависимости от дня недели')
plt.xlabel('День недели (0 - понедельник, 6 - воскресенье)')
plt.ylabel('Количество поездок')
plt.savefig('day_of_week_trips.png')
print("График сохранен в day_of_week_trips.png")

# Определение аномальных дней
# Анализ количества поездок по датам
date_counts = train_df['pickup_datetime'].dt.date.value_counts().sort_index()
date_counts_df = pd.DataFrame({'date': date_counts.index, 'count': date_counts.values})

# Находим аномальные дни (дни с наименьшим количеством поездок)
anomalous_dates = date_counts_df.sort_values('count').head(2)['date'].values
print(f"Аномальные дни: {anomalous_dates}")

# Модифицированная функция create_features
def create_features(data_frame):
    # Добавление дня недели и бинарного признака для аномальных дней
    X = pd.concat([
        data_frame.pickup_datetime.apply(lambda x: x.timetuple().tm_yday),  # день в году
        data_frame.pickup_datetime.apply(lambda x: x.hour),  # час
        data_frame.pickup_datetime.apply(lambda x: x.weekday()),  # день недели (Задание 3)
        data_frame.pickup_datetime.apply(lambda x: 1 if x.date() in anomalous_dates else 0)  # бинарный признак для аномальных дней (Задание 2)
    ], axis=1, keys=['day', 'hour', 'weekday', 'anomalous_day'])
    
    return X, data_frame.log_trip_duration

# Создание признаков
print("Создание признаков...")
X_train, y_train = create_features(train_df)
X_test, y_test = create_features(test_df)

# Задание 4: Масштабирование вещественного признака (день в году)
print("Масштабирование признаков...")
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled['day'] = scaler.fit_transform(X_train[['day']])
X_test_scaled['day'] = scaler.transform(X_test[['day']])

# Обучение Lasso регрессии
print("Обучение Lasso регрессии...")
lasso = Lasso(alpha=2.65e-05)
lasso.fit(X_train_scaled, y_train)

# Оценка качества
y_pred = lasso.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")

# Подсчет отобранных признаков
coef_threshold = 1e-6
selected_features = np.sum(np.abs(lasso.coef_) > coef_threshold)
print(f"Количество отобранных признаков: {selected_features}")

# One-hot кодирование
print("Применение one-hot кодирования...")
X_train_onehot = pd.get_dummies(X_train, columns=['hour', 'weekday'], drop_first=False)
X_test_onehot = pd.get_dummies(X_test, columns=['hour', 'weekday'], drop_first=False)

# Масштабирование вещественного признака после one-hot кодирования
X_train_onehot['day'] = scaler.fit_transform(X_train_onehot[['day']])
X_test_onehot['day'] = scaler.transform(X_test_onehot[['day']])

print(f"Количество признаков после one-hot кодирования: {X_train_onehot.shape[1]}")

# Обучение Lasso регрессии на данных с one-hot кодированием
lasso_onehot = Lasso(alpha=2.65e-05)
lasso_onehot.fit(X_train_onehot, y_train)

# Оценка качества
y_pred_onehot = lasso_onehot.predict(X_test_onehot)
mse_onehot = mean_squared_error(y_test, y_pred_onehot)
print(f"MSE после one-hot кодирования: {mse_onehot}")

# Подсчет отобранных признаков
selected_features_onehot = np.sum(np.abs(lasso_onehot.coef_) > coef_threshold)
print(f"Количество отобранных признаков после one-hot кодирования: {selected_features_onehot}")