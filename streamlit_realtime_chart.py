import streamlit as st
import numpy as np
import time

# 시뮬레이션을 위한 변수 정의
num_points = 100
x_min, x_max = 0, 10
y_min, y_max = 0, 1

# 그래프 생성
fig = st.line_chart(np.zeros(num_points))

# 무한루프 시작
while True:
    # 데이터 생성
    x = np.linspace(x_min, x_max, num_points)
    y = np.random.uniform(y_min, y_max, size=(num_points,))
    
    # 그래프 업데이트
    fig.add_rows(new_data=[y])
    
    # 0.1초 대기
    time.sleep(0.1)
