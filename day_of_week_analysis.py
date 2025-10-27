import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_csv('train.zip', compression='zip', header=0, sep=',', quotechar='"')

# Преобразование строковых дат в datetime
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

# Извлечение дня недели (0 - понедельник, 6 - воскресенье)
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek

# Создание словаря для названий дней недели
day_names = {
    0: 'Понедельник',
    1: 'Вторник',
    2: 'Среда',
    3: 'Четверг',
    4: 'Пятница',
    5: 'Суббота',
    6: 'Воскресенье'
}

# Преобразование числовых значений дней недели в названия
df['day_name'] = df['day_of_week'].map(day_names)

# Построение графика
plt.figure(figsize=(12, 6))
sns.countplot(x='day_name', data=df, order=[day_names[i] for i in range(7)])
plt.title('Количество поездок в зависимости от дня недели')
plt.xlabel('День недели')
plt.ylabel('Количество поездок')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('day_of_week_trips.png')
plt.show()