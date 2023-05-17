import streamlit as st
import numpy as np
import pandas as pd
import glob #파일 다 읽어줌
import os, sys
import matplotlib.pyplot as plt

#os.chdir("C:\\Users\\San\\Desktop\\23-1\\cap\\practice\\샘플 데이터\\NASA Bearing Dataset\\2nd_test")
os.chdir("C:/Users/San/Desktop/23-1/cap/practice/샘플 데이터/NASA Bearing Dataset/2nd_test/2nd_test")
# os.chdir=change directory로 작업중인 디렉토리를 변경하는 함수
file_list = glob.glob("**/*.39", recursive=True)
#"**/*.39": 이 패턴은 현재 디렉토리 및 모든 하위 디렉토리에서 확장자가 .39인 파일을 찾습니다.
#"**/"는 현재 디렉토리 및 모든 하위 디렉토리를 의미합니다.
#"*.39"는 확장자가 .39인 파일을 의미합니다.
#recursive=True: 이 옵션은 모든 하위 디렉토리에서도 파일을 검색하도록 합니다.
#glove함수는 읽어주는 함수

#len(file_list) # 행의 길이(2번째 테스트 데이터)

df = pd.read_csv(file_list[0], sep='\t',header=None) #file_list[0]에 있는걸 읽어오는 행 파일이 tab으로 구분되어있음. 
# 첫번째 파일을 불러오는 것
#df.head() # head로 앞에서부터 5개의 값을 뽑음
df
#이렇게 뽑으면 너무 많은(20479*948)데이터가 생기므로 신호처리를 해줘야 한다

#x와y열을 나눠줘야 밑에 시각화 그래프를 그릴 수 있다.

x = df.index  # 데이터의 행(row) 번호를 x값으로 사용
#y = df['a']   # 'a' 열(가속도 정보)을 y값으로 사용
y = df.iloc[:, 1:4]  # 두 번째 열부터 네 번째 열까지는 가속도 정보
#x는 데이터의 갯수(rows값) y는 열 값의 범위

#plt.figure(figsize=(12,4)) # 사이즈 지정
#plt.plot(df)
#plt.show()

# 그래프 생성
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(x, y)
st.pyplot(fig)

# 그래프 출력
# plt.show()