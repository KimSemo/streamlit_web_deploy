# import streamlit as st
# import numpy as np
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt
# import time

# # 데이터 가져오기
# def get_realtime_data(file_name):
#     base_url = "https://raw.githubusercontent.com/changyeon99/TimingBeltData/main/"
#     url = base_url + file_name
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # 오류가 발생하면 예외 발생
#         data = response.text
#         return data
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error retrieving data: {str(e)}")
#         return None

# # 신호 처리 함수 - Raw
# def process_raw_data(df):
#     return df

# # 신호 처리 함수 - RMS
# def process_rms_data(df):
#     processed_df = pd.DataFrame()
#     for column in df.columns:
#         processed_column = df[column].dropna()  # NaN 값 제거
#         rms_value = np.sqrt(np.mean(np.square(processed_column)))
#         processed_df[column] = [rms_value]  # RMS 값 추가
#     return processed_df

# # 신호 처리 함수 - FFT
# def process_fft_data(df):
#     processed_df = pd.DataFrame()
#     for column in df.columns:
#         fft_values = np.fft.fft(df[column])
#         processed_df[column] = np.abs(fft_values)
#     return processed_df

# # 신호 처리 기법 선택
# processing_technique = st.sidebar.selectbox("Signal Processing Technique", ["Raw", "RMS", "FFT"])

# # 스트림릿 앱 시작
# st.header("Real-time Data Visualization")

# # 버튼 클릭 여부 확인
# button_clicked = st.sidebar.button("Show Graph")

# # 그래프 카운터
# graph_counter = 0

# # 데이터 처리 및 시각화
# start_time = time.strptime("2023-05-28 00:03", "%Y-%m-%d %H:%M")
# end_time = time.strptime("2023-05-28 10:46", "%Y-%m-%d %H:%M")
# current_time = start_time

# while current_time <= end_time:
#     file_name = time.strftime("A%Y-%m-%d %H:%M.txt", current_time)
#     data = get_realtime_data(file_name)
#     if data and button_clicked:
#         # 데이터 전처리
#         lines = data.strip().split("\n")
#         values_1 = []
#         values_2 = []
#         values_3 = []
#         for line in lines:
#             if line:
#                 parts = line.split(",")
#                 if len(parts) >= 3:
#                     try:
#                         value_1 = float(parts[0])
#                         value_2 = float(parts[1])
#                         value_3 = float(parts[2])
#                         values_1.append(value_1)
#                         values_2.append(value_2)
#                         values_3.append(value_3)
#                     except ValueError:
#                         continue

#         # 데이터프레임 생성
#         df = pd.DataFrame({"Value 1": values_1, "Value 2": values_2, "Value 3": values_3})

#         # 신호 처리 기법 적용
#         if processing_technique == "Raw":
#             processed_df = process_raw_data(df)
#         elif processing_technique == "RMS":
#             processed_df = process_rms_data(df)
#         elif processing_technique == "FFT":
#             processed_df = process_fft_data(df)

#         # 그래프 생성
#         if not processed_df.empty:  # 데이터프레임이 비어 있지 않은 경우에만 그래프 생성
#             fig, ax = plt.subplots(figsize=(12, 4))
#             for column in processed_df.columns:
#                 ax.plot(processed_df.index, processed_df[column], label=column)
#             ax.set_xlabel("Time")
#             ax.set_ylabel("Value")
#             ax.set_title(f"Real-time Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#             ax.legend()
#             st.pyplot(fig)
#             graph_counter += 1

#         # 그래프 출력 제한
#         if graph_counter >= 10:
#             break

#         # 데이터 출력
#         if not processed_df.empty:  # 데이터프레임이 비어 있지 않은 경우에만 데이터 출력
#             st.dataframe(processed_df)

#     current_time = time.localtime(time.mktime(current_time) + 60)  # 1분씩 증가
#     time.sleep(1)  # 1초 대기

# import streamlit as st
# import numpy as np
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt
# import time

# # 데이터 가져오기
# def get_realtime_data(file_name):
#     base_url = "https://raw.githubusercontent.com/changyeon99/TimingBeltData/main/"
#     url = base_url + file_name
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # 오류가 발생하면 예외 발생
#         data = response.text
#         return data
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error retrieving data: {str(e)}")
#         return None

# # 신호 처리 함수 - Raw
# def process_raw_data(df):
#     return df

# # 신호 처리 함수 - RMS
# def process_rms_data(df):
#     processed_df = pd.DataFrame()
#     for column in df.columns:
#         processed_column = df[column].dropna()  # NaN 값 제거
#         processed_df[column] = np.sqrt(np.mean(np.square(processed_column)))
#     return processed_df

# # 신호 처리 함수 - FFT
# def process_fft_data(df):
#     processed_df = pd.DataFrame()
#     for column in df.columns:
#         fft_values = np.fft.fft(df[column])
#         processed_df[column] = np.abs(fft_values)
#     return processed_df

# # 스케일 조정 함수 - Min-Max 스케일링
# def min_max_scaling(df):
#     return (df - df.min()) / (df.max() - df.min())

# # 신호 스케일링 함수
# def scale_signal(df):
#     scaled_df = pd.DataFrame()
#     for column in df.columns:
#         scaled_df[column] = min_max_scaling(df[column])
#     return scaled_df

# # 신호 처리 기법 선택
# processing_technique = st.sidebar.selectbox("Signal Processing Technique", ["Raw", "RMS", "FFT"])

# # 스트림릿 앱 시작
# st.header("Real-time Data Visualization")

# # 버튼 클릭 여부 확인
# button_clicked = st.sidebar.button("Show Graph")

# # 데이터 처리 및 시각화
# start_time = time.strptime("2023-05-28 00:03", "%Y-%m-%d %H:%M")
# end_time = time.strptime("2023-05-28 10:46", "%Y-%m-%d %H:%M")
# current_time = start_time

# while current_time <= end_time:
#     file_name = time.strftime("A%Y-%m-%d %H:%M.txt", current_time)
#     data = get_realtime_data(file_name)
#     if data and button_clicked:
#         # 데이터 전처리
#         lines = data.strip().split("\n")
#         values_1 = []
#         values_2 = []
#         values_3 = []
#         for line in lines:
#             if line:
#                 parts = line.split(",")
#                 if len(parts) >= 3:
#                     try:
#                         value_1 = float(parts[0])
#                         value_2 = float(parts[1])
#                         value_3 = float(parts[2])
#                         values_1.append(value_1)
#                         values_2.append(value_2)
#                         values_3.append(value_3)
#                     except ValueError:
#                         continue

#         # 데이터프레임 생성
#         df = pd.DataFrame({"Value 1": values_1, "Value 2": values_2, "Value 3": values_3})

#         # 신호 처리 기법 적용
#         if processing_technique == "Raw":
#             processed_df = process_raw_data(df)
#         elif processing_technique == "RMS":
#             processed_df = process_rms_data(df)
#         elif processing_technique == "FFT":
#             processed_df = process_fft_data(df)

#         # 스케일 조정
#         processed_df = scale_signal(processed_df)

#         # 그래프 생성
#         fig, ax = plt.subplots(figsize=(12, 4))
#         for column in processed_df.columns:
#             ax.plot(processed_df.index, processed_df[column], label=column)
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Scaled Value")
#         ax.set_title(f"Real-time Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#         ax.legend()
#         st.pyplot(fig)

#         # 그래프 삭제
#         plt.close(fig)

#         # 데이터 출력
#         st.dataframe(processed_df)

#     # 다음 시간으로 업데이트
#     current_time = time.localtime(time.mktime(current_time) + 60)  # 60 seconds = 1 minute

#     # 1초마다 데이터 갱신
#     time.sleep(1)

import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import time

# 데이터 가져오기
def get_realtime_data(file_name):
    base_url = "https://raw.githubusercontent.com/changyeon99/TimingBeltData/main/"
    url = base_url + file_name
    try:
        response = requests.get(url)
        response.raise_for_status()  # 오류가 발생하면 예외 발생
        data = response.text
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Error retrieving data: {str(e)}")
        return None

# 신호 처리 함수 - Raw
def process_raw_data(df):
    processed_df = pd.DataFrame(index=df.index)  # 인덱스 설정
    for column in df.columns:
        processed_column = df[column].dropna()  # NaN 값 제거
        processed_df[column] = (processed_column - processed_column.min()) / (processed_column.max() - processed_column.min())
    return processed_df

# 신호 처리 함수 - RMS
def process_rms_data(df):
    processed_df = pd.DataFrame(index=df.index)  # 인덱스 설정
    for column in df.columns:
        processed_column = df[column].dropna()  # NaN 값 제거
        processed_df[column] = (processed_column - processed_column.mean()) / processed_column.std()
    return processed_df

# 신호 처리 함수 - FFT
def process_fft_data(df):
    processed_df = pd.DataFrame(index=df.index)  # 인덱스 설정
    for column in df.columns:
        fft_values = np.fft.fft(df[column])
        processed_df[column] = np.abs(fft_values)
    return processed_df

# 신호 처리 기법 선택
processing_technique = st.sidebar.selectbox("Signal Processing Technique", ["Raw", "RMS", "FFT"])

# 스트림릿 앱 시작
st.header("Real-time Data Visualization")

# 버튼 클릭 여부 확인
button_clicked = st.sidebar.button("Show Graph")

# 데이터 처리 및 시각화
start_time = time.strptime("2023-05-28 00:03", "%Y-%m-%d %H:%M")
end_time = time.strptime("2023-05-28 10:46", "%Y-%m-%d %H:%M")
current_time = start_time

fig, ax = plt.subplots(figsize=(12, 4))
graphs = []  # 그래프 객체를 저장하는 리스트

while current_time <= end_time:
    file_name = time.strftime("A%Y-%m-%d %H:%M.txt", current_time)
    data = get_realtime_data(file_name)
    if data and button_clicked:
        # 데이터 전처리
        lines = data.strip().split("\n")
        values_1 = []
        values_2 = []
        values_3 = []
        for line in lines:
            if line:
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        value_1 = float(parts[0])
                        value_2 = float(parts[1])
                        value_3 = float(parts[2])
                        values_1.append(value_1)
                        values_2.append(value_2)
                        values_3.append(value_3)
                    except ValueError:
                        continue

        # 데이터프레임 생성
        df = pd.DataFrame({"Value 1": values_1, "Value 2": values_2, "Value 3": values_3})

        # 신호 처리 기법 적용
        if processing_technique == "Raw":
            processed_df = process_raw_data(df)
        elif processing_technique == "RMS":
            processed_df = process_rms_data(df)
        elif processing_technique == "FFT":
            processed_df = process_fft_data(df)

        # 그래프 생성
        ax.clear()  # 이전 그래프 삭제
        for column in processed_df.columns:
            ax.plot(processed_df.index, processed_df[column], label=column)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title(f"Real-time Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
        ax.legend()
        graph = ax.figure  # 현재 그래프 객체 저장
        graphs.append(graph)  # 그래프 리스트에 추가

        # 그래프 출력
        st.pyplot(graph)

        # 데이터 출력
        st.dataframe(processed_df)

        # 그래프 개수 확인
        if len(graphs) >= 10:
            old_graph = graphs.pop(0)  # 가장 오래된 그래프 삭제
            plt.close(old_graph)  # 그래프 창 닫기

    # 다음 시간으로 업데이트
    current_time = time.localtime(time.mktime(current_time) + 60)  # 60 seconds = 1 minute

    # 1초마다 데이터 갱신
    time.sleep(1)

# 그래프 창 닫기
for graph in graphs:
    plt.close(graph)







