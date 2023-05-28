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
import glob
import os, sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from scipy.fft import fft

os.chdir("C:\\Users\\San\\Desktop\\23-1\\cap\\practice\\샘플 데이터\\NASA Bearing Dataset\\2nd_test")
file_list = glob.glob("**/*.39", recursive=True)

df = pd.read_csv(file_list[0], sep='\t', header=None)

# FFT 신호 처리 함수
def fft_signal(stats):
    return np.abs(fft(stats, axis=0))

# 2000개의 데이터 샘플을 하나의 파일로 추출
fft_arr = []
for file in file_list:
    df = pd.read_csv(file, sep='\t', header=None)
    fft_value = fft_signal(df.values)
    fft_arr.append(fft_value)
fft_arr = np.concatenate(fft_arr, axis=0)

fft_df = pd.DataFrame(fft_arr, columns=['ch1', 'ch2', 'ch3', 'ch4'])
fft_df.to_csv('C:\\Users\\San\\Desktop\\23-1\\cap\\practice\\샘플 데이터\\NASA Bearing Dataset\\FFT_bearing.csv', index=None)

# 스트림릿 앱의 헤더 및 설명 추가
st.header("FFT 신호 처리")
st.write("FFT를 사용하여 신호를 처리하고 시각화합니다.")


# Matplotlib figure 생성
with st.spinner('로딩 중...'):
    fig, ax = plt.subplots(figsize=(12, 8), tight_layout=True)

    # 전체 그래프 데이터
    all_signals = []

    # 개별 그래프 플롯
    for i in range(4):
        signal = fft_df[f'ch{i+1}'].values

        # 선 그래프 플롯
        ax.plot(signal, label=f'Ch{i+1} FFT')

        # 전체 그래프 데이터에 추가
        all_signals.extend(signal)

    # 전체 그래프 플롯
    ax.plot(all_signals, label='Combined FFT', color='black', linewidth=2)

    # 범례와 제목 추가
    ax.legend()
    ax.set_title('Combined FFT')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Amplitude')

    # 그래프 표시
    st.pyplot(fig)




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
