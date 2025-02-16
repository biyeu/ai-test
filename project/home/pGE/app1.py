import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
st.set_page_config(
    page_icon="🌍",
)
st.title('Top 10 quốc gia có dân số cao nhất')
st.sidebar.title('Home')
st.sidebar.title('Some chart')
st.sidebar.title('Predictions')


# Load the CSV file
data = pd.read_csv('world_population.csv')

# Draw a bar chart of the top 10 countries by 2022 Population
sorted_data = data.sort_values(by='Rank')
st.bar_chart(sorted_data.set_index('Country/Territory')['2022 Population'].iloc[:10])

ef = pd.DataFrame(data)
ef = ef.sort_values(by='Rank')

st.area_chart(ef.set_index('Country/Territory')['2022 Population'].iloc[:10])

st.title('Các yếu tố ảnh hưởng đến dân số')
st.write('1. Tốc độ tăng trưởng')
st.write('2. Diên tích') 
st.write('3. Tỉ trọng (per km²)')
st.write('4. Tỷ lệ dân số thế giới')
st.title('Các biểu đồ về các yếu tố ảnh hưởng đến dân số')

a = st.selectbox('Chọn yếu tố', ['Growth Rate','Area (km²)','Density (per km²)','World Population Percentage'])
st.write(f'Chart about top 10 countries has most {a} in 2022')
chart = st.bar_chart(ef.set_index('Country/Territory')[a].iloc[:10])
line = st.line_chart(ef.set_index('Country/Territory')[a].iloc[:10])

st.title('Chi tiết tương quan âm/dương của từng thuộc tính')
# Create a correlation matrix
correlation_matrix = ef[['Growth Rate', 'Area (km²)', 'Density (per km²)', 'World Population Percentage']].corr()

# Display the correlation matrix
st.write(correlation_matrix)

ffu = pd.DataFrame(correlation_matrix)



# Draw a bar chart of the correlation matrix
st.bar_chart(ffu)

# st.write("""
#  Vậy ta thấy rằng: yếu tố cần thiết nhất để dự đoán dân số của một quốc gia là Tốc độ tăng trưởng
# """)

