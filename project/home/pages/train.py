import pandas as pd
import numpy as np
# import seaborn as sns # Vẽ heatmap
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # chia tập train và test
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('world_population.csv')
df = pd.DataFrame(data)
df.head(9)
data = df.sort_values(by='Rank')

X = data[['Growth Rate', 'Area (km²)', 'Density (per km²)', 'World Population Percentage']]
y = data['2020 Population']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)