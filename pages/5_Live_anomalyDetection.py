import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 데이터 스트림 받아오기
def get_realtime_data():
    url = "https://raw.githubusercontent.com/username/repo/main/realtime_data.csv"  # GitHub에서 데이터 파일의 URL을 입력합니다.
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return None
    except:
        return None

# rms 신호추출기법 root mean square 신호의 크기를 추출
def rms(stats):
    return np.sqrt(np.mean(stats**2, axis=0))

# 데이터 처리 및 신호처리 함수
def process_data(df):
    # 신호 처리
    rms_value = rms(df.values)

    # RMS 데이터 출력
    print("RMS Data:")
    print(rms_value)

    # RMS 데이터 시각화
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(rms_value)), rms_value)
    ax.set_xticks(range(len(rms_value)))
    ax.set_xticklabels(["Ch1", "Ch2", "Ch3"])
    ax.set_ylabel("RMS Value")
    ax.set_title("RMS Data")
    plt.show()

    # 이상감지 수행
    cl = 0.2  # 임계값 설정
    integrated_tsScore = np.sqrt(np.sum((tsScore_arr - trScore_arr)**2, axis=1))
    outidx = np.where(integrated_tsScore > cl)[0]

    # 이상감지 시각화
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(integrated_tsScore, color='blue')
    ax.axhline(y=cl, color='red')

    for idx in outidx:
        ax.axvline(x=idx, color='red', linestyle='-', alpha=0.1)

    ax.set_title("Anomaly Detection")
    plt.show()

# RMS 데이터 파일 경로
rms_file_path = 'C:\\Users\\San\\Desktop\\23-1\\cap\\practice\\샘플 데이터\\NASA Bearing Dataset\\RMS_bearing.csv'

# Train Data, Test Data
rms_df = pd.read_csv(rms_file_path)
trdat = rms_df.values[0:400, :]
tsdat = rms_df.values

trScore_arr = np.zeros([trdat.shape[0], trdat.shape[1]])
tsScore_arr = np.zeros([tsdat.shape[0], trdat.shape[1]])

lr = LinearRegression()

input_idx = np.arange(trdat.shape[1]).tolist()

for idx in input_idx:
    input_idx = np.arange(trdat.shape[1]).tolist()
    input_idx.remove(idx)

    lr.fit(trdat[:, input_idx], trdat[:, idx])
    trScore = lr.predict(trdat[:, input_idx])
    tsScore = lr.predict(tsdat[:, input_idx])

    trScore_arr[:, idx] = trScore
    tsScore_arr[:, idx] = tsScore

# 실시간 데이터 수신 및 처리
while True:
    # GitHub에서 실시간 데이터 가져오기
    data = get_realtime_data()
    if data:
        # 데이터 전처리
        df_realtime = pd.read_csv(data)

        # 데이터 처리 및 신호 처리
        process_data(df_realtime)

    else:
        print("Failed to retrieve real-time data from GitHub.")


# 위 코드 실행 안될 시 사용 데이터 수집, 신호처리, 이상감지 묶은 코드 URL이랑 데이터 저장 위치,이름 바꿔주기
# import requests
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# import streamlit as st

# # 데이터 스트림 받아오기
# def get_realtime_data():
#     url = "https://raw.githubusercontent.com/username/repo/main/realtime_data.csv"  # GitHub에서 데이터 파일의 URL을 입력합니다.
#     try:
#         response = requests.get(url)
#         if response.status_code == 200:
#             return response.text
#         else:
#             return None
#     except:
#         return None

# # rms 신호추출기법 root mean square 신호의 크기를 추출
# def rms(stats):
#     return np.sqrt(np.mean(stats**2, axis=0))

# # 데이터 처리 및 신호처리 함수
# def process_data(df):
#     # 신호 처리
#     rms_value = rms(df.values)

#     # RMS 데이터 출력
#     st.write("RMS Data:")
#     st.write(rms_value)

#     # RMS 데이터 시각화
#     fig, ax = plt.subplots(figsize=(12, 4))
#     ax.bar(range(len(rms_value)), rms_value)
#     ax.set_xticks(range(len(rms_value)))
#     ax.set_xticklabels(["Ch1", "Ch2", "Ch3"])
#     ax.set_ylabel("RMS Value")
#     ax.set_title("RMS Data")
#     st.pyplot(fig)

#     # 이상감지 수행
#     cl = 0.2  # 임계값 설정
#     integrated_tsScore = np.sqrt(np.sum((tsScore_arr - trScore_arr)**2, axis=1))
#     outidx = np.where(integrated_tsScore > cl)[0]

#     # 이상감지 시각화
#     fig, ax = plt.subplots(figsize=(12, 4))
#     ax.plot(integrated_tsScore, color='blue')
#     ax.axhline(y=cl, color='red')

#     for idx in outidx:
#         ax.axvline(x=idx, color='red', linestyle='-', alpha=0.1)

#     ax.set_title("Anomaly Detection")
#     st.pyplot(fig)

# # RMS 데이터 파일 경로
# rms_file_path = 'C:\\Users\\San\\Desktop\\23-1\\cap\\practice\\샘플 데이터\\NASA Bearing Dataset\\RMS_bearing.csv'

# # Train Data, Test Data
# rms_df = pd.read_csv(rms_file_path)
# trdat = rms_df.values[0:400, :]
# tsdat = rms_df.values

# trScore_arr = np.zeros([trdat.shape[0], trdat.shape[1]])
# tsScore_arr = np.zeros([tsdat.shape[0], trdat.shape[1]])

# lr = LinearRegression()

# input_idx = np.arange(trdat.shape[1]).tolist()

# # 스트림릿 애플리케이션 시작
# st.title("Real-time Anomaly Detection")

# while True:
#     # GitHub에서 실시간 데이터 가져오기
#     data = get_realtime_data()
#     if data:
#         # 데이터 전처리
#         df_realtime = pd.read_csv(data)

#         # 데이터 처리 및 신호 처리
#         process_data(df_realtime)

#     else:
#         st.write("Failed to retrieve real-time data from GitHub.")
