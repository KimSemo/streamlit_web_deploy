import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st

# 데이터 스트림 받아오기
def get_realtime_data():
    url = "https://raw.githubusercontent.com/changyeon99/TimingBeltData/main/realtime_data.csv"  # GitHub에서 데이터 파일의 URL을 입력합니다.
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

# fft 신호처리 함수
def fft(signal):
    fft_values = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal))
    return freqs, fft_values

# 데이터 처리 및 신호처리 함수
def process_data(df, process_rms, process_fft):
    # RMS 신호처리
    if process_rms:
        rms_value = rms(df.values)

        # RMS 데이터 출력
        st.write("RMS Data:")
        st.write(rms_value)

        # RMS 데이터 시각화
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(len(rms_value)), rms_value)
        ax.set_xticks(range(len(rms_value)))
        ax.set_xticklabels(["Ch1", "Ch2", "Ch3"])
        ax.set_ylabel("RMS Value")
        ax.set_title("RMS Data")
        st.pyplot(fig)

    # FFT 신호처리
    if process_fft:
        for i in range(3):
            signal = df[f'ch{i+1}'].values
            freqs, fft_values = fft(signal)

            # FFT 데이터 시각화
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(freqs, np.abs(fft_values))
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Ch{i+1} FFT')
            st.pyplot(fig)

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

# 스트림릿 애플리케이션 시작
st.title("Real-time Anomaly Detection")

# 체크박스 옵션 선택
process_rms = st.checkbox("RMS 신호처리")
process_fft = st.checkbox("FFT 신호처리")

while True:
    # GitHub에서 실시간 데이터 가져오기
    data = get_realtime_data()
    if data:
        # 데이터 전처리
        df_realtime = pd.read_csv(data)

        # 데이터 처리 및 신호 처리
        process_data(df_realtime, process_rms, process_fft)

    else:
        st.write("Failed to retrieve real-time data from GitHub.")
