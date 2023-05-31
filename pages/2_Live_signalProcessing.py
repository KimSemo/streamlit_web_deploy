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

#         # 신호 처리 기법 적용
#         if processing_technique == "Raw":
#             processed_df = process_raw_data(df)
#         elif processing_technique == "RMS":
#             processed_df = process_rms_data(df)
#         elif processing_technique == "FFT":
#             processed_df = process_fft_data(df)

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

#############위 코드가 현재 배포한 코드 밑에 코드는 데이터 받아오는 방법 바꿔서 

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
#         if processing_technique == "Raw":
#             processed_df = process_raw_data(df)
#         elif processing_technique == "RMS":
#             processed_df = process_rms_data(df)
#         elif processing_technique == "FFT":
#             processed_df = process_fft_data(df)

#         # 기존 데이터프레임과 새로운 데이터프레임 합치기
#         df_combined = pd.concat([df_combined, processed_df], ignore_index=True)

#         # 그래프 그리기
#         ax.clear()
#         for column in processed_df.columns:
#             ax.plot(df_combined.index, df_combined[column], label=column)
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

############이 밑 코드는 신호처리 부분까지 데이터가 갱신될때 마다 갱신된 데이터와 기존데이터를 합쳐서 신호처리 하도록

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

# # 신호 처리 함수 - RMS
# def process_rms_data(df):
#     processed_df = pd.DataFrame(index=df.index)  # 인덱스 설정
#     for column in df.columns:
#         processed_column = df[column].dropna()  # NaN 값 제거
#         rms_values = np.sqrt(np.mean(processed_column ** 2))  # RMS 계산
#         processed_column /= rms_values  # RMS 값으로 나누어 정규화
#         scaler = StandardScaler()
#         processed_column = scaler.fit_transform(processed_column.values.reshape(-1, 1))[:, 0]  # 스케일링
#         processed_df[column] = processed_column
#     return processed_df

# # # 신호 처리 함수 - FFT
# # def process_fft_data(df):
# #     processed_df = pd.DataFrame(index=df.index[:-1])  # 인덱스 설정 (마지막 데이터 제외)
# #     for column in df.columns:
# #         fft_values = np.fft.fft(df[column][1:])  # 첫 번째 데이터 제외하고 FFT 적용
# #         processed_values = np.abs(fft_values)
# #         processed_values[processed_values < 1e-10] = 0  # 아주 작은 값은 0으로 처리
# #         processed_df[column] = processed_values
# #     return processed_df

# # 신호 처리 함수 - FFT
# def process_fft_data(df):
#     processed_df = pd.DataFrame(index=df.index[:-1])  # 인덱스 설정 (마지막 데이터 제외)
#     for column in df.columns:
#         fft_values = np.fft.fft(df[column][1:])  # 첫 번째 데이터 제외하고 FFT 적용
#         processed_values = np.abs(fft_values)
#         processed_values[processed_values < 1e-10] = 0  # 아주 작은 값은 0으로 처리

#         # Value1, Value2, Value3에서 특정 범위 이상의 값을 제거하는 부분 추가
#         if column in ["Value 1", "Value 2", "Value 3"]:
#             threshold = 0.08  # Value1, Value2, Value3에서 제거할 임계값 설정
#             processed_values[processed_values > threshold] = 0

#         processed_df[column] = processed_values
#     return processed_df



# # 데이터 전처리 함수
# def preprocess_data(df):
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(df)
#     processed_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
#     return processed_df

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
#         if processing_technique == "Raw":
#             processed_df = df.copy()
#         elif processing_technique == "RMS":
#             processed_df = process_rms_data(df)
#         elif processing_technique == "FFT":
#             processed_df = process_fft_data(df)

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

# # # 신호 처리 함수 - RMS
# # def process_rms_data(df):
# #     processed_df = pd.DataFrame(index=df.index)  # 인덱스 설정
# #     window_size = 5  # 이동 평균에 사용할 윈도우 크기 설정
# #     for column in df.columns:
# #         processed_column = df[column].dropna()  # NaN 값 제거
# #         rms_values = np.sqrt(np.mean(processed_column ** 2))  # RMS 계산
# #         processed_column /= rms_values  # RMS 값으로 나누어 정규화
# #         processed_column = processed_column.rolling(window_size, min_periods=1).mean()  # 이동 평균 적용
# #         scaler = StandardScaler()
# #         processed_column = scaler.fit_transform(processed_column.values.reshape(-1, 1))[:, 0]  # 스케일링
# #         processed_df[column] = processed_column
# #     return processed_df

# # # 신호 처리 함수 - FFT
# # def process_fft_data(df):
# #     processed_df = pd.DataFrame(index=df.index[:-1])  # 인덱스 설정 (마지막 데이터 제외)
# #     window_size = 5  # 이동 평균에 사용할 윈도우 크기 설정
# #     for column in df.columns:
# #         fft_values = np.fft.fft(df[column][1:])  # 첫 번째 데이터 제외하고 FFT 적용
# #         processed_values = np.abs(fft_values)
# #         processed_values[processed_values < 1e-10] = 0  # 아주 작은 값은 0으로 처리
# #         processed_values = pd.Series(processed_values).rolling(window_size, min_periods=1).mean()  # 이동 평균 적용
# #         processed_df[column] = processed_values
# #     return processed_df

# # 신호 처리 함수 - RMS
# def process_rms_data(df):
#     processed_df = pd.DataFrame(index=df.index)  # 인덱스 설정
#     window_size = 5  # 이동 평균에 사용할 윈도우 크기 설정
#     zscore_threshold = 3  # Z-score 이상치 탐지 임계값 설정

#     for column in df.columns:
#         processed_column = df[column].dropna()  # NaN 값 제거
#         rms_values = np.sqrt(np.mean(processed_column ** 2))  # RMS 계산
#         processed_column /= rms_values  # RMS 값으로 나누어 정규화
#         processed_column = processed_column.rolling(window_size, min_periods=1).mean()  # 이동 평균 적용
#         scaler = StandardScaler()
#         processed_column = scaler.fit_transform(processed_column.values.reshape(-1, 1))[:, 0]  # 스케일링

#         # 이상치 탐지 및 처리
#         z_scores = np.abs((processed_column - processed_column.mean()) / processed_column.std())
#         processed_column[z_scores > zscore_threshold] = np.nan

#         processed_df[column] = processed_column

#     return processed_df

# # 신호 처리 함수 - FFT with FFT shift
# def process_fft_data(df):
#     processed_df = pd.DataFrame(index=df.index[:-1])  # 인덱스 설정 (마지막 데이터 제외)
#     window_size = 5  # 이동 평균에 사용할 윈도우 크기 설정
#     zscore_threshold = 3  # Z-score 이상치 탐지 임계값 설정

#     for column in df.columns:
#         fft_values = np.fft.fft(df[column][1:])  # 첫 번째 데이터 제외하고 FFT 적용

#         # FFT shift 적용
#         fft_values_shifted = np.fft.fftshift(fft_values)

#         # FFT 결과 정규화
#         fft_values_normalized = np.abs(fft_values_shifted) / len(fft_values_shifted)

#         # 이동 평균 적용
#         processed_values = pd.Series(fft_values_normalized).rolling(window_size, min_periods=1).mean()

#         # 이동 평균 값이 0보다 작으면 0으로 처리
#         processed_values[processed_values < 0] = 0

#         # 이동 평균 값을 정규화
#         scaler = StandardScaler()
#         processed_values = scaler.fit_transform(processed_values.values.reshape(-1, 1))[:, 0]

#         # 이상치 탐지 및 처리
#         z_scores = np.abs((processed_values - processed_values.mean()) / processed_values.std())
#         processed_values[z_scores > zscore_threshold] = np.nan

#         processed_df[column] = processed_values

#     return processed_df







# # 데이터 전처리 함수
# def preprocess_data(df):
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(df)
#     processed_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
#     return processed_df

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
#         if processing_technique == "Raw":
#             processed_df = df.copy()
#         elif processing_technique == "RMS":
#             processed_df = process_rms_data(df)
#         elif processing_technique == "FFT":
#             processed_df = process_fft_data(df)

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

# 신호 처리 함수 - FFT with FFT shift
def process_fft_data(df):
    processed_df = pd.DataFrame(index=df.index[:-1])  # 인덱스 설정 (마지막 데이터 제외)
    window_size = 5  # 이동 평균에 사용할 윈도우 크기 설정
    zscore_threshold = 3  # Z-score 이상치 탐지 임계값 설정

    for column in df.columns:
        fft_values = np.fft.fft(df[column][1:])  # 첫 번째 데이터 제외하고 FFT 적용

        # FFT shift 적용
        fft_values_shifted = np.fft.fftshift(fft_values)

        # FFT 결과 정규화
        fft_values_normalized = np.abs(fft_values_shifted) / len(fft_values_shifted)

        # 이동 평균 적용
        processed_values = pd.Series(fft_values_normalized).rolling(window_size, min_periods=1).mean()

        # 이동 평균 값이 0보다 작으면 0으로 처리
        processed_values[processed_values < 0] = 0

        # 이동 평균 값을 정규화
        scaler = StandardScaler()
        processed_values = scaler.fit_transform(processed_values.values.reshape(-1, 1))[:, 0]

        # 이상치 탐지 및 처리
        z_scores = np.abs((processed_values - processed_values.mean()) / processed_values.std())
        processed_values[z_scores > zscore_threshold] = np.nan

        processed_df[column] = processed_values

    return processed_df

# # 신호 처리 함수 - FFT with FFT shift and Windowing
# def process_fft_data(df):
#     processed_df = pd.DataFrame(index=df.index[1:-1])  # Adjusted index to match the length of processed FFT values
#     window_size = 5  # 이동 평균에 사용할 윈도우 크기 설정
#     zscore_threshold = 3  # Z-score 이상치 탐지 임계값 설정

#     for column in df.columns:
#         # Apply data trimming
#         trimmed_data = df[column][1:-1]  # Exclude first and last data points

#         # Apply Hann window
#         window = np.hanning(len(trimmed_data))
#         windowed_data = trimmed_data * window

#         fft_values = np.fft.fft(windowed_data)  # FFT 적용

#         # FFT shift 적용
#         fft_values_shifted = np.fft.fftshift(fft_values)

#         # FFT 결과 정규화
#         fft_values_normalized = np.abs(fft_values_shifted) / len(fft_values_shifted)

#         # Add processed values to the DataFrame
#         processed_df[column] = fft_values_normalized

#     return processed_df




# 신호 처리 함수 - 이동 평균
def process_moving_average_data(df):
    processed_df = pd.DataFrame(index=df.index)  # 인덱스 설정
    window_size = 5  # 이동 평균에 사용할 윈도우 크기 설정
    zscore_threshold = 3  # Z-score 이상치 탐지 임계값 설정

    for column in df.columns:
        processed_column = df[column].dropna()  # NaN 값 제거
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

# 신호 처리 기법 선택
processing_technique = st.sidebar.selectbox("Signal Processing Technique", ["Raw", "RMS", "FFT", "Moving Average"])

# 스트림릿 앱 시작
st.header("Real-time Data Visualization")

# 버튼 클릭 여부 확인
button_clicked = st.sidebar.button("Show Graph")

# 데이터 처리 및 시각화
start_time = time.strptime("2023-05-29 19:50", "%Y-%m-%d %H:%M")
end_time = time.strptime("2023-05-29 23:49", "%Y-%m-%d %H:%M")
current_time = start_time

#fig, ax = plt.subplots(figsize=(12, 4))
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
        elif processing_technique == "FFT":
            processed_df = process_fft_data(df)
        elif processing_technique == "Moving Average":
            processed_df = process_moving_average_data(df)

        # 데이터 스케일링
        if processing_technique != "Raw":
            processed_df = preprocess_data(processed_df)

        # 기존 데이터프레임과 새로운 데이터프레임 합치기
        df_combined = pd.concat([df_combined, processed_df], ignore_index=True)

        # 그래프 그리기
        # ax.clear()
        # if not df_combined.empty:
        #     for column in df_combined.columns:
        #         ax.plot(df_combined.index, df_combined[column], label=column)
        # else:
        #     for column in df.columns:
        #         ax.plot(df.index, df[column], label=column)
        # ax.set_xlim(0, len(df_combined))
        # ax.set_ylim(df_combined.min().min(), df_combined.max().max())
        # ax.set_xlabel("Time")
        # ax.set_ylabel("Value")
        # ax.set_title(f"Real-time Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
        # ax.legend()
        # st.pyplot(fig)

        # Update the line chart with new rows
        chart.add_rows(processed_df)

    # 다음 시간으로 업데이트
    current_time = time.localtime(time.mktime(current_time) + 60)  # 60 seconds = 1 minute

    # 1초마다 데이터 갱신
    time.sleep(1)

# 그래프 창 닫기
#plt.close(fig)











