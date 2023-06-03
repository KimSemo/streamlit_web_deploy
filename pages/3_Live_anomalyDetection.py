# import streamlit as st
# import numpy as np
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt
# import time
# from sklearn.preprocessing import StandardScaler

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

# # 신호 처리 함수 - 이동 평균
# def process_moving_average_data(df):
#     processed_df = pd.DataFrame(index=df.index)  # 인덱스 설정
#     window_size = 5  # 이동 평균에 사용할 윈도우 크기 설정
#     zscore_threshold = 3  # Z-score 이상치 탐지 임계값 설정

#     for column in df.columns:
#         processed_column = df[column].dropna()  # NaN 값 제거
#         processed_column = processed_column.rolling(window_size, min_periods=1).mean()  # 이동 평균 적용
#         scaler = StandardScaler()
#         processed_column = scaler.fit_transform(processed_column.values.reshape(-1, 1))[:, 0]  # 스케일링

#         # 이상치 탐지 및 처리
#         z_scores = np.abs((processed_column - processed_column.mean()) / processed_column.std())
#         processed_column[z_scores > zscore_threshold] = np.nan

#         processed_df[column] = processed_column

#     return processed_df

# # 데이터 전처리 함수
# def preprocess_data(df):
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(df)
#     processed_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
#     return processed_df

# # 이상 감지 함수
# def anomaly_detection(df, trScore_arr, tsScore_arr):
#     cl = 0.2  # 임계값 설정

#     # 이상 감지 수행
#     integrated_tsScore = np.sqrt(np.sum((tsScore_arr - trScore_arr) ** 2, axis=1))
#     outidx = np.where(integrated_tsScore > cl)[0]

#     # 이상 감지 시각화
#     fig, ax = plt.subplots(figsize=(12, 4))
#     ax.plot(integrated_tsScore, color='blue')
#     ax.axhline(y=cl, color='red')

#     for idx in outidx:
#         ax.axvline(x=idx, color='red', linestyle='-', alpha=0.1)

#     ax.set_title("Anomaly Detection")
#     plt.show()

# # 신호 처리 기법 선택
# processing_technique = "Moving Average"

# # 스트림릿 앱 시작
# st.header("Real-time Data Visualization")

# # 버튼 클릭 여부 확인
# button_clicked = st.sidebar.button("Show Graph")

# # 데이터 처리 및 시각화
# start_time = time.strptime("2023-05-28 00:03", "%Y-%m-%d %H:%M")
# end_time = time.strptime("2023-05-28 10:46", "%Y-%m-%d %H:%M")
# current_time = start_time

# fig, ax = plt.subplots(figsize=(12, 4))
# df_combined = pd.DataFrame()  # 이전 데이터와 새로운 데이터를 합칠 데이터프레임

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

#         # 선택한 신호 처리 기법 적용
#         if processing_technique == "Moving Average":
#             processed_df = process_moving_average_data(df)

#             # 이상 감지 수행
#             anomaly_detection(processed_df, processed_df.values[:-1], processed_df.values[1:])

#         # 데이터 스케일링
#         if processing_technique != "Raw":
#             processed_df = preprocess_data(processed_df)

#         # 기존 데이터프레임과 새로운 데이터프레임 합치기
#         df_combined = pd.concat([df_combined, processed_df], ignore_index=True)

#         # 그래프 그리기
#         ax.clear()
#         if not df_combined.empty:
#             for column in df_combined.columns:
#                 ax.plot(df_combined.index, df_combined[column], label=column)
#         else:
#             for column in df.columns:
#                 ax.plot(df.index, df[column], label=column)
#         ax.set_xlim(0, len(df_combined))
#         ax.set_ylim(df_combined.min().min(), df_combined.max().max())
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Value")
#         ax.set_title(f"Real-time Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#         ax.legend()
#         st.pyplot(fig)

#     # 다음 시간으로 업데이트
#     current_time = time.localtime(time.mktime(current_time) + 60)  # 60 seconds = 1 minute

#     # 1초마다 데이터 갱신
#     time.sleep(1)

# # 그래프 창 닫기
# plt.close(fig)


##############################Z-score threshold 이용 이상감지
import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler

# 데이터 가져오기
def get_realtime_data(file_name):
    base_url = "https://raw.githubusercontent.com/changyeon99/TimingBeltData_CMPS/main/"
    url = base_url + file_name
    try:
        response = requests.get(url)
        response.raise_for_status()  # 오류가 발생하면 예외 발생
        data = response.text
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Error retrieving data: {str(e)}")
        return None

# 신호 처리 함수 - RMS
def process_rms_data(df):
    processed_df = pd.DataFrame(index=df.index)  # 인덱스 설정
    window_size = 5  # 이동 평균에 사용할 윈도우 크기 설정
    zscore_threshold = 3  # Z-score 이상치 탐지 임계값 설정

    for column in df.columns:
        processed_column = df[column].dropna()  # NaN 값 제거
        rms_values = np.sqrt(np.mean(processed_column ** 2))  # RMS 계산
        processed_column /= rms_values  # RMS 값으로 나누어 정규화
        processed_column = processed_column.rolling(window_size, min_periods=1).mean()  # 이동 평균 적용
        scaler = StandardScaler()
        processed_column = scaler.fit_transform(processed_column.values.reshape(-1, 1))[:, 0]  # 스케일링

        # 이상치 탐지 및 처리
        z_scores = np.abs((processed_column - processed_column.mean()) / processed_column.std())
        processed_column[z_scores > zscore_threshold] = np.nan

        processed_df[column] = processed_column

    return processed_df

# 데이터 전처리 함수
def preprocess_data(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    processed_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    return processed_df

# 이상치 탐지 함수
def detect_anomalies(df, threshold=3):
    anomalies_df = pd.DataFrame(index=df.index)  # 인덱스 설정

    for column in df.columns:
        values = df[column].dropna()  # NaN 값 제거

        # 이상치 탐지
        z_scores = np.abs((values - values.mean()) / values.std())
        anomalies = z_scores > threshold
        anomalies_df[column] = values.where(anomalies, np.nan)

    return anomalies_df

# 신호 처리 기법 선택
processing_technique = st.sidebar.selectbox("Signal Processing Technique", ["Raw", "RMS"])

# 이상치 탐지 임계값 설정
zscore_threshold = st.sidebar.slider("Z-Score Threshold", min_value=1, max_value=5, value=3)

# 스트림릿 앱 시작
st.header("Real-time Data Visualization")

# 버튼 클릭 여부 확인
button_clicked = st.sidebar.button("Show Graph")

# 데이터 처리 및 시각화
start_time = time.strptime("2023-05-29 19:50", "%Y-%m-%d %H:%M")
end_time = time.strptime("2023-05-29 23:49", "%Y-%m-%d %H:%M")
current_time = start_time

df_combined = pd.DataFrame()  # 이전 데이터와 새로운 데이터를 합칠 데이터프레임

chart = st.line_chart()  # Create an empty line chart

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

        # 선택한 신호 처리 기법 적용
        if processing_technique == "Raw":
            processed_df = df.copy()
        elif processing_technique == "RMS":
            processed_df = process_rms_data(df)

        # 데이터 스케일링
        if processing_technique != "Raw":
            processed_df = preprocess_data(processed_df)

        # 이상치 탐지
        anomalies_df = detect_anomalies(processed_df, zscore_threshold)

        # 기존 데이터프레임과 새로운 데이터프레임 합치기
        df_combined = pd.concat([df_combined, anomalies_df], ignore_index=True)

        # 그래프 그리기
        chart.add_rows(anomalies_df)

    # 다음 시간으로 업데이트
    current_time = time.localtime(time.mktime(current_time) + 60)  # 60 seconds = 1 minute

    # 1초마다 데이터 갱신
    time.sleep(1)

