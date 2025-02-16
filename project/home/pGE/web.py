import streamlit as st
import pandas as pd

st.title('Dự đoán dân số của cac quốc gia')

st.sidebar.title('Home')
st.sidebar.title('Some chart','app1.py')
st.sidebar.title('Predictions')

st.write('Bảng dữ liệu dân số thế giới')
data = pd.read_csv('world_population.csv')
ef = pd.DataFrame(data)
ef = ef.sort_values(by='Rank')
st.write(ef)
st.title('Chưc năng')
st.write('01.Vẽ biểu đồ cột của 10 quốc gia có dân số cao nhất năm 2022')
st.write('02.Xem Xem phân tích tập dữ liệu dân số thế giới')
st.write('03.Dự đoán dân số của các quốc gia bằng AI')