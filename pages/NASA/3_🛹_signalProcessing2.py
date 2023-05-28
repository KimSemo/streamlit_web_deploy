# import streamlit as st
# import numpy as np
# import pandas as pd
# import glob
# import os, sys
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('agg')

# os.chdir("C:\\Users\\San\\Desktop\\23-1\\cap\\practice\\샘플 데이터\\NASA Bearing Dataset\\2nd_test")
# file_list = glob.glob("**/*.39", recursive=True)

# df = pd.read_csv(file_list[0], sep='\t', header=None)

# # FFT 신호 처리 함수
# def fft(signal):
#     spectrum = np.fft.fft(signal)
#     freq = np.fft.fftfreq(len(signal))
#     return spectrum, freq

# # RMS 신호 처리 함수
# def rms(stats):
#     return np.sqrt(np.mean(stats**2, axis=0))

# # 2000개의 데이터 샘플을 하나의 파일로 추출
# rms_arr = np.array([])
# for file in file_list:
#     df = pd.read_csv(file, sep='\t', header=None)
#     rms_value = rms(df.values)
#     rms_arr = np.concatenate([rms_arr, rms_value], axis=0)
# rms_arr = rms_arr.reshape(len(file_list), 4)

# rms_df = pd.DataFrame(rms_arr, columns=['ch1', 'ch2', 'ch3', 'ch4'])
# rms_df.to_csv('C:\\Users\\San\\Desktop\\23-1\\cap\\practice\\샘플 데이터\\NASA Bearing Dataset\\RMS_bearing.csv', index=None)

# # 스트림릿 앱의 헤더 및 설명 추가
# st.header("FFT 신호 처리")
# st.write("FFT를 사용하여 신호를 처리하고 시각화합니다.")

# # Matplotlib figure 생성
# with st.spinner('로딩 중...'):
#     fig, axs = plt.subplots(4, 1, figsize=(12, 12))
#     for i in range(4):
#         signal = rms_df[f'ch{i+1}'].values
#         spectrum, freq = fft(signal)

#         axs[i].plot(freq, np.abs(spectrum))
#         axs[i].set_title(f'Ch{i+1} FFT Spectrum')
#         axs[i].set_xlabel('Frequency')
#         axs[i] 스트림릿 앱에 표시
# st.pyplot(.set_ylabel('Amplitude')

# # 플롯을fig)

import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy.fft import fft
import time

# 데이터 가져오기
def get_realtime_data(file_name):
    base_url = "https://raw.githubusercontent.com/changyeon99/TimingBeltData/main/"
    url = base_url + file_name
    response = requests.get(url)
    if response.status_code == 200:
        data = response.text
        return data
    else:
        return None

# RMS 신호처리 함수
def rms(stats):
    return np.sqrt(np.mean(stats**2, axis=0))

# FFT 신호처리 함수
def fft_signal(stats):
    return np.abs(fft(stats, axis=0))

# 스트림릿 앱 시작
st.header("Real-time Data Visualization")
st.subheader("Select Signal Processing Technique")

# 신호처리 기법 선택
processing_technique = st.selectbox("Processing Technique", ["RMS", "FFT"])

# 데이터 처리 및 시각화
start_time = time.strptime("2023-05-28 00:03", "%Y-%m-%d %H:%M")
end_time = time.strptime("2023-05-28 10:46", "%Y-%m-%d %H:%M")
current_time = start_time

while current_time <= end_time:
    file_name = time.strftime("A%Y-%m-%d %H:%M.txt", current_time)
    data = get_realtime_data(file_name)
    if data:
        # 데이터 전처리
        lines = data.strip().split("\n")
        values_1 = []
        values_2 = []
        values_3 = []

        for line in lines:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                if parts[0] != '' and parts[1] != '' and parts[2] != '':
                    try:
                        value_1 = float(parts[0])
                        value_2 = float(parts[1])
                        value_3 = float(parts[2])
                        values_1.append(value_1)
                        values_2.append(value_2)
                        values_3.append(value_3)
                    except ValueError:
                        # 부동 소수점으로 변환할 수 없는 문자열이 있는 경우 건너뜁니다.
                        continue

        df = pd.DataFrame({"ch1": values_1, "ch2": values_2, "ch3": values_3})

        # 신호처리 기법 선택에 따른 처리
        if processing_technique == "RMS":
            processed_values = rms(df.values)
            processed_df = pd.DataFrame(processed_values.reshape(1, -1), columns=[f"RMS Value {i+1}" for i in range(processed_values.shape[0])])
        else:
            processed_values = fft_signal(df.values)
            processed_df = pd.DataFrame(processed_values.T, columns=["FFT Value 1", "FFT Value 2", "FFT Value 3"])

        # 그래프 출력
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(processed_df.values.flatten())
        ax.legend(processed_df.columns)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title(f"{processing_technique} Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
        st.pyplot(fig)

        # 데이터 출력
        st.dataframe(processed_df)

    # 다음 시간으로 업데이트
    current_time = time.localtime(time.mktime(current_time) + 60)  # 60 seconds = 1 minute

    # 그래프 닫기
    plt.close()

    # 1초마다 데이터 갱신
    time.sleep(1)






# # Matplotlib figure 생성
# with st.spinner('로딩 중...'):
#     fig, axs = plt.subplots(5, 1, figsize=(12, 16))

#     # 각각의 그래프 플롯
#     for i in range(4):
#         signal = fft_df[f'ch{i+1}'].values

#         axs[i].plot(signal)
#         axs[i].set_title(f'Ch{i+1} FFT')
#         axs[i].set_xlabel('Frequency')
#         axs[i].set_ylabel('Amplitude')

#     # 4개의 결과를 합친 선 그래프 플롯
#     combined_signal = np.sum(fft_df.values, axis=1)
#     axs[4].plot(combined_signal, color='red')
#     axs[4].set_title('Combined FFT')
#     axs[4].set_xlabel('Frequency')
#     axs[4].set_ylabel('Amplitude')

#     # 그래프 간격 조정
#     plt.tight_layout()

#     # 플롯을 스트림릿 앱에 표시
#     st.pyplot(fig)






# # Matplotlib figure 생성
# with st.spinner('로딩 중...'):
#     fig, axs = plt.subplots(4, 1, figsize=(12, 12))
#     for i in range(4):
#         signal = fft_df[f'ch{i+1}'].values

#         axs[i].plot(signal)
#         axs[i].set_title(f'Ch{i+1} FFT')
#         axs[i].set_xlabel('Frequency')
#         axs[i].set_ylabel('Amplitude')

# # 플롯을 스트림릿 앱에 표시
# st.pyplot(fig)


# 실시간수집시, fft신호처리시 사용 코드
# import streamlit as st
# import pandas as pd
# import numpy as np
# import requests
# import io
# import matplotlib.pyplot as plt
# from scipy.fft import fft

# # GitHub raw 데이터 URL
# raw_data_url = "https://raw.githubusercontent.com/your_username/your_repository/main/data.csv"

# # 데이터 로드 및 전처리 함수
# def load_data(url):
#     response = requests.get(url)
#     response.raise_for_status()
#     content = response.content.decode("utf-8")
#     df = pd.read_csv(io.StringIO(content))
#     return df

# # 데이터 저장 및 업데이트 함수
# def update_data():
#     data_df = load_data(raw_data_url)
#     data_df.to_csv("data.csv", index=False)  # 데이터를 로컬에 저장
#     return data_df

# # FFT 신호 처리 함수
# def fft_signal(stats):
#     return np.abs(fft(stats, axis=0))

# # 데이터 로드 또는 업데이트
# data_df = update_data()

# # FFT 처리
# fft_arr = fft_signal(data_df.values)

# # FFT 결과 데이터프레임 생성
# fft_df = pd.DataFrame(fft_arr, columns=data_df.columns)

# # 스트림릿 앱의 헤더 및 설명 추가
# st.header("Real-time Data Visualization")
# st.write("Data loaded from GitHub and visualized in real-time.")

# # Matplotlib figure 생성
# fig, axs = plt.subplots(len(fft_df.columns), 1, figsize=(12, 6 * len(fft_df.columns)))

# # 시각화
# for i, col in enumerate(fft_df.columns):
#     signal = fft_df[col].values

#     axs[i].plot(signal)
#     axs[i].set_title(f"{col} FFT")
#     axs[i].set_xlabel("Frequency")
#     axs[i].set_ylabel("Amplitude")

# # 플롯을 스트림릿 앱에 표시
# st.pyplot(fig)
