import streamlit as st
import pandas as pd

st.title('Dự đoán dân số của cac quốc gia')

st.sidebar.title("Navigation")
st.sidebar.button('Home')
st.sidebar.button('Some chart')
st.sidebar.button('Predictions')
if st.sidebar.button("Home"):
    st.switch_page("pages/web.py")

if st.sidebar.button("Chart"):
    st.switch_page("pages/app.py")

if st.sidebar.button("Predictions"):
    st.switch_page("pages/predict.py")

st.write('Bảng dữ liệu dân số thế giới')
data = pd.read_csv('population_by_country_2020.csv')
ef = pd.DataFrame(data)
ef = ef.sort_values(by='Rank')
st.write(ef)
st.title('Chưc năng')
st.write('01.Vẽ biểu đồ cột của 10 quốc gia có dân số cao nhất năm 2022')
st.write('02.Xem Xem phân tích tập dữ liệu dân số thế giới')
st.write('03.Dự đoán dân số của các quốc gia bằng AI')
