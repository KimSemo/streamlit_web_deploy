# import requests
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression

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
#     print("RMS Data:")
#     print(rms_value)

#     # RMS 데이터 시각화
#     fig, ax = plt.subplots(figsize=(12, 4))
#     ax.bar(range(len(rms_value)), rms_value)
#     ax.set_xticks(range(len(rms_value)))
#     ax.set_xticklabels(["Ch1", "Ch2", "Ch3"])
#     ax.set_ylabel("RMS Value")
#     ax.set_title("RMS Data")
#     plt.show()

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
#     plt.show()

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

# for idx in input_idx:
#     input_idx = np.arange(trdat.shape[1]).tolist()
#     input_idx.remove(idx)

#     lr.fit(trdat[:, input_idx], trdat[:, idx])
#     trScore = lr.predict(trdat[:, input_idx])
#     tsScore = lr.predict(tsdat[:, input_idx])

#     trScore_arr[:, idx] = trScore
#     tsScore_arr[:, idx] = tsScore

# # 실시간 데이터 수신 및 처리
# while True:
#     # GitHub에서 실시간 데이터 가져오기
#     data = get_realtime_data()
#     if data:
#         # 데이터 전처리
#         df_realtime = pd.read_csv(data)

#         # 데이터 처리 및 신호 처리
#         process_data(df_realtime)

#     else:
#         print("Failed to retrieve real-time data from GitHub.")


# import streamlit as st
# import numpy as np
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt
# import time

# # Function to retrieve real-time data
# def get_realtime_data(file_name):
#     base_url = "https://raw.githubusercontent.com/changyeon99/TimingBeltData/main/"
#     url = base_url + file_name
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Raise exception for any errors
#         data = response.text
#         return data
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error retrieving data: {str(e)}")
#         return None

# # Function to process raw data
# def process_raw_data(df):
#     processed_df = pd.DataFrame(index=df.index)
#     for column in df.columns:
#         processed_column = df[column].dropna()
#         processed_df[column] = (processed_column - processed_column.min()) / (processed_column.max() - processed_column.min())
#     return processed_df

# # Function to process RMS data
# def process_rms_data(df):
#     processed_df = pd.DataFrame(index=df.index)
#     for column in df.columns:
#         processed_column = df[column].dropna()
#         processed_df[column] = (processed_column - processed_column.min()) / (processed_column.max() - processed_column.min())
#     return processed_df

# # Function to process FFT data
# def process_fft_data(df):
#     processed_df = pd.DataFrame(index=df.index)
#     for column in df.columns:
#         fft_values = np.fft.fft(df[column])
#         processed_df[column] = np.abs(fft_values)
#     return processed_df

# # Anomaly detection function
# def detect_anomalies(df):
#     # Perform anomaly detection logic here
#     # You can use statistical methods, machine learning models, or any other technique

#     # Example: Threshold-based anomaly detection
#     threshold = 0.5
#     anomalies = df > threshold

#     return anomalies

# # Signal processing technique selection
# processing_technique = st.sidebar.selectbox("Signal Processing Technique", ["Raw", "RMS", "FFT"])

# # Start Streamlit app
# st.header("Real-time Data Visualization")

# # Check if the button is clicked
# button_clicked = st.sidebar.button("Show Graph")

# # Data processing and visualization
# start_time = time.strptime("2023-05-28 00:03", "%Y-%m-%d %H:%M")
# end_time = time.strptime("2023-05-28 10:46", "%Y-%m-%d %H:%M")
# current_time = start_time

# fig, ax = plt.subplots(figsize=(12, 4))
# graphs = []

# while current_time <= end_time:
#     file_name = time.strftime("A%Y-%m-%d %H:%M.txt", current_time)
#     data = get_realtime_data(file_name)
#     if data and button_clicked:
#         # Data preprocessing
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

#         # Create DataFrame
#         df = pd.DataFrame({"Value 1": values_1, "Value 2": values_2, "Value 3": values_3})

#         # Apply signal processing technique
#         if processing_technique == "Raw":
#             processed_df = process_raw_data(df)
#         elif processing_technique == "RMS":
#             processed_df = process_rms_data(df)
#         elif processing_technique == "FFT":
#             processed_df = process_fft_data(df)

#         # Anomaly detection
#         anomalies = detect_anomalies(processed_df)

#         # Generate graph
#         ax.clear()
#         for column in processed_df.columns:
#             ax.plot(processed_df.index, processed_df[column], label=column)
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Value")
#         ax.set_title(f"Real-time Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#         ax.legend()
#         graph = ax.figure
#         graphs.append(graph)

#         # Display graph
#         st.pyplot(graph)

#         # Display processed data
#         st.dataframe(processed_df)

#         # Display anomalies
#         st.write("Anomalies:")
#         st.dataframe(anomalies)

#         # Check the number of graphs
#         if len(graphs) >= 10:
#             old_graph = graphs.pop(0)
#             plt.close(old_graph)

#     # Update to the next time
#     current_time = time.localtime(time.mktime(current_time) + 60)

#     # Refresh data every second
#     time.sleep(1)

# # Close graph windows
# for graph in graphs:
#     plt.close(graph)

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
#     processed_df = pd.DataFrame(index=df.index)  # 인덱스 설정
#     for column in df.columns:
#         processed_column = df[column].dropna()  # NaN 값 제거
#         processed_df[column] = (processed_column - processed_column.min()) / (processed_column.max() - processed_column.min())
#     return processed_df

# # 신호 처리 함수 - RMS
# def process_rms_data(df):
#     processed_df = pd.DataFrame(index=df.index)  # 인덱스 설정
#     for column in df.columns:
#         processed_column = df[column].dropna()  # NaN 값 제거
#         processed_df[column] = (processed_column - processed_column.min()) / (processed_column.max() - processed_column.min())
#         #processed_df[column] = (processed_column - processed_column.mean()) / processed_column.std()
#     return processed_df

# # 신호 처리 함수 - FFT
# def process_fft_data(df):
#     processed_df = pd.DataFrame(index=df.index)  # 인덱스 설정
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

# # 기존 데이터를 저장할 데이터프레임
# combined_df = pd.DataFrame()

# # 데이터 처리 및 시각화
# start_time = time.strptime("2023-05-28 00:03", "%Y-%m-%d %H:%M")
# end_time = time.strptime("2023-05-28 10:46", "%Y-%m-%d %H:%M")
# current_time = start_time

# fig, ax = plt.subplots(figsize=(12, 4))
# graphs = []  # 그래프 객체를 저장하는 리스트

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

#         # 기존 데이터와 합치기
#         combined_df = pd.concat([combined_df, df])

#         # 신호 처리 기법 적용
#         if processing_technique == "Raw":
#             processed_df = process_raw_data(combined_df)
#         elif processing_technique == "RMS":
#             processed_df = process_rms_data(combined_df)
#         elif processing_technique == "FFT":
#             processed_df = process_fft_data(combined_df)

#         # 그래프 생성
#         ax.clear()  # 이전 그래프 삭제
#         for column in processed_df.columns:
#             ax.plot(processed_df.index, processed_df[column], label=column)
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Value")
#         ax.set_title(f"Real-time Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#         ax.legend()
#         graph = ax.figure  # 현재 그래프 객체 저장
#         graphs.append(graph)  # 그래프 리스트에 추가

#         # 그래프 출력
#         st.pyplot(graph)

#         # 데이터 출력
#         st.dataframe(processed_df)

#         # 그래프 개수 확인
#         if len(graphs) >= 10:
#             old_graph = graphs.pop(0)  # 가장 오래된 그래프 삭제
#             plt.close(old_graph)  # 그래프 창 닫기

#     # 다음 시간으로 업데이트
#     current_time = time.localtime(time.mktime(current_time) + 60)  # 60 seconds = 1 minute

#     # 1초마다 데이터 갱신
#     time.sleep(1)

# # 그래프 창 닫기
# for graph in graphs:
#     plt.close(graph)


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


################################################## 여기서 부터 찐 코드 이상감지 테스트 시작