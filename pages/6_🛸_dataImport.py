import streamlit as st
import numpy as np
import pandas as pd
import glob #파일 다 읽어줌
import os, sys
import matplotlib.pyplot as plt


os.chdir("C:\\Users\\San\\Desktop\\23-1\\cap\\practice\\샘플 데이터\\NASA Bearing Dataset\\2nd_test")

# os.chdir=change directory로 작업중인 디렉토리를 변경하는 함수
file_list = glob.glob("**/*.39", recursive=True)

#"**/*.39": 이 패턴은 현재 디렉토리 및 모든 하위 디렉토리에서 확장자가 .39인 파일을 찾습니다.
#"**/"는 현재 디렉토리 및 모든 하위 디렉토리를 의미합니다.
#"*.39"는 확장자가 .39인 파일을 의미합니다.
#recursive=True: 이 옵션은 모든 하위 디렉토리에서도 파일을 검색하도록 합니다.
#glove함수는 읽어주는 함수

#len(file_list) # 행의 길이(2번째 테스트 데이터)

df = pd.read_csv(file_list[0], sep='\t',header=None) #file_list[0]에 있는걸 읽어오는 행 파일이 tab으로 구분되어있음. 

# 데이터프레임 출력
st.header("Uploaded Data")
st.write(df)

# 데이터 처리
x = df.index
y = df.iloc[:, 1:4]

# 그래프 생성
st.header("Data Visualization")
fig, ax = plt.subplots(figsize=(12, 4))
for column in y.columns:
    ax.plot(x, y[column], label=column)
ax.legend()

# 그래프 출력
st.pyplot(fig)



## 깃허브를 통해 실시간 데이터 받는 코드!

# import streamlit as st
# import numpy as np
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt


# # GitHub 저장소 정보
# github_user = "changyeon99"
# repo_name = "forCapstone"
# file_name = "data.txt"

# # 데이터 가져오기
# def get_realtime_data():
#     url = f"https://raw.githubusercontent.com/changyeon99/forCapstone/main/{file_name}"
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.text
#         return data
#     else:
#         return None

# # 스트림릿 앱 시작
# st.header("Real-time Data Visualization")

# # 데이터 처리 및 시각화
# while True:
#     data = get_realtime_data()
#     if data:
#         # 데이터 전처리
#         lines = data.strip().split("\n")
#         values = [float(line) for line in lines]

#         # 데이터프레임 생성
#         df = pd.DataFrame({"Value": values})

#         # 그래프 생성
#         fig, ax = plt.subplots(figsize=(12, 4))
#         ax.plot(df.index, df["Value"])
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Value")
#         ax.set_title("Real-time Data")
#         st.pyplot(fig)

#     # 1초마다 데이터 갱신
#     time.sleep(1)

