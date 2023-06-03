# 깃허브를 통해 실시간 데이터 받는 코드!

# import streamlit as st
# import numpy as np
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt
# import time



# # 데이터 가져오기
# def get_realtime_data():
#     url = f"https://raw.githubusercontent.com/changyeon99/TimingBeltData/main/A2023-05-28 00:03.txt"
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


## 위에가 original



# ######잘 돌아가는 코드 근데 파일갱신이 필요
# import streamlit as st
# import numpy as np
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt
# import time

# # 데이터 가져오기
# def get_realtime_data():
#     url = "https://raw.githubusercontent.com/changyeon99/TimingBeltData/main/A2023-05-28%2000:03.txt"
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
#         values = []
#         for line in lines:
#             if line:
#                 try:
#                     parts = line.split(",")
#                     value = float(parts[0])  # 센서값이 두 번째 열에 위치하는 것으로 가정
#                     values.append(value)
#                 except ValueError:
#                     continue

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


########위에 잘 돌아가는 코드 수정해봄(데이터 파일 갱신되게)

## start시간 파일만 맞으면 끝나는 시간을 엄청 뒤로해두면 실시간으로 임포트한 데이터를 볼 수 있다.
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
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.text
#         return data
#     else:
#         return None
# df = pd.read_txt(file_list[0], sep='\t',header=None)
# # 스트림릿 앱 시작
# st.header("Real-time Data Visualization")

# # 데이터 처리 및 시각화
# start_time = time.strptime("2023-05-28 00:03", "%Y-%m-%d %H:%M")
# end_time = time.strptime("2023-05-28 11:28", "%Y-%m-%d %H:%M")       ##10:46
# current_time = start_time

# while current_time <= end_time:
#     file_name = time.strftime("A%Y-%m-%d %H:%M.txt", current_time)
#     data = get_realtime_data(file_name)
#     if data:
#         # 데이터 전처리 및 시각화 작업 수행
#         lines = data.strip().split("\n")
#         values = []
#         for line in lines:
#             if line:
#                 try:
#                     value = float(line.split(",")[0])
#                     values.append(value)
#                 except ValueError:
#                     continue

#         if values:
#             df = pd.DataFrame({"Value": values})

#             fig, ax = plt.subplots(figsize=(12, 4))
#             ax.plot(df.index, df["Value"])
#             ax.set_xlabel("Time")
#             ax.set_ylabel("Value")
#             ax.set_title(f"Real-time Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#             st.pyplot(fig)

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
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.text
#         return data
#     else:
#         return None

# # 스트림릿 앱 시작
# st.header("Real-time Data Visualization")

# # 데이터 처리 및 시각화
# start_time = time.strptime("2023-05-28 00:03", "%Y-%m-%d %H:%M")
# end_time = time.strptime("2023-05-28 10:46", "%Y-%m-%d %H:%M")
# current_time = start_time

# while current_time <= end_time:
#     file_name = time.strftime("A%Y-%m-%d %H:%M.txt", current_time)
#     data = get_realtime_data(file_name)
#     if data:
#         # 데이터 전처리
#         lines = data.strip().split("\n")
#         values = [float(line.split(",")[0]) for line in lines if line]

#         # 데이터프레임 생성
#         df = pd.DataFrame({"Value": values})

#         # 그래프 생성
#         fig, ax = plt.subplots(figsize=(12, 4))
#         ax.plot(df.index, df["Value"])
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Value")
#         ax.set_title(f"Real-time Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#         st.pyplot(fig)

#         # 데이터 출력
#         st.dataframe(df)

#     # 다음 시간으로 업데이트
#     current_time = time.localtime(time.mktime(current_time) + 60)  # 60 seconds = 1 minute

#     # 1초마다 데이터 갱신
#     time.sleep(1)




################## 최종 완료 코드 ###################
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
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.text
#         return data
#     else:
#         return None

# # 스트림릿 앱 시작
# st.header("Real-time Data Visualization")

# # 데이터 처리 및 시각화
# start_time = time.strptime("2023-05-28 00:03", "%Y-%m-%d %H:%M")
# end_time = time.strptime("2023-05-28 10:46", "%Y-%m-%d %H:%M")
# current_time = start_time

# while current_time <= end_time:
#     file_name = time.strftime("A%Y-%m-%d %H:%M.txt", current_time)
#     data = get_realtime_data(file_name)
#     if data:
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

#         # 그래프 생성
#         fig, ax = plt.subplots(figsize=(12, 4))
#         ax.plot(df.index, df["Value 1"], label="Value 1")
#         ax.plot(df.index, df["Value 2"], label="Value 2")
#         ax.plot(df.index, df["Value 3"], label="Value 3")
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Value")
#         ax.set_title(f"Real-time Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#         ax.legend()
#         st.pyplot(fig)

#         # 데이터 출력
#         st.dataframe(df)

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
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.text
#         return data
#     else:
#         return None

# # 스트림릿 앱 시작
# st.header("Real-time Data Visualization")

# # 데이터 처리 및 시각화
# start_time = time.strptime("2023-05-28 00:03", "%Y-%m-%d %H:%M")
# end_time = time.strptime("2023-05-28 10:46", "%Y-%m-%d %H:%M")
# current_time = start_time

# while current_time <= end_time:
#     file_name = time.strftime("A%Y-%m-%d %H:%M.txt", current_time)
#     data = get_realtime_data(file_name)
#     if data:
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

#         # 그래프 생성
#         fig, ax = plt.subplots(figsize=(12, 4))
#         ax.plot(df.index, df["Value 1"], label="Value 1")
#         ax.plot(df.index, df["Value 2"], label="Value 2")
#         ax.plot(df.index, df["Value 3"], label="Value 3")
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Value")
#         ax.set_title(f"Real-time Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#         ax.legend()
#         st.pyplot(fig)

#         # 그래프 삭제
#         plt.close(fig)

#         # 데이터 출력
#         st.dataframe(df)

#     # 다음 시간으로 업데이트
#     current_time = time.localtime(time.mktime(current_time) + 60)  # 60 seconds = 1 minute

#     # 1초마다 데이터 갱신
#     time.sleep(1)



#############################위에 코드가 기존 시각화 코드!!!!!
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
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.text
#         return data
#     else:
#         return None

# # 스트림릿 앱 시작
# st.header("Real-time Data Visualization")

# # 그래프 프레임 및 축 설정
# fig, ax = plt.subplots(figsize=(12, 4))
# ax.set_xlabel("Time")
# ax.set_ylabel("Value")
# ax.set_title("Real-time Data")
# df_combined = pd.DataFrame()

# # 데이터 처리 및 시각화
# start_time = time.strptime("2023-05-28 00:03", "%Y-%m-%d %H:%M")
# end_time = time.strptime("2023-05-28 10:46", "%Y-%m-%d %H:%M")
# current_time = start_time

# while current_time <= end_time:
#     file_name = time.strftime("A%Y-%m-%d %H:%M.txt", current_time)
#     data = get_realtime_data(file_name)
#     if data:
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

#         # 기존 데이터프레임과 새로운 데이터프레임 합치기
#         df_combined = pd.concat([df_combined, df], ignore_index=True)

#         # 그래프 그리기
#         ax.clear()
#         ax.plot(df_combined.index, df_combined["Value 1"], label="Value 1")
#         ax.plot(df_combined.index, df_combined["Value 2"], label="Value 2")
#         ax.plot(df_combined.index, df_combined["Value 3"], label="Value 3")
#         ax.set_xlim(0, len(df_combined))
#         ax.set_ylim(df_combined.min().min(), df_combined.max().max())
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Value")
#         ax.set_title("Real-time Data")
#         ax.legend()
#         st.pyplot(fig)

#     # 다음 시간으로 업데이트
#     current_time = time.localtime(time.mktime(current_time) + 60)  # 60 seconds = 1 minute

#     # 1초마다 데이터 갱신
#     time.sleep(1)

############그래프가 계속 뜨게 한 이유가 뭐냐면



# import streamlit as st
# import numpy as np
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

# # 정규화 함수
# def normalize_data(df):
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(df)
#     scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
#     return scaled_df

# # 표준화 함수
# def standardize_data(df):
#     scaler = StandardScaler()
#     standardized_data = scaler.fit_transform(df)
#     standardized_df = pd.DataFrame(standardized_data, columns=df.columns, index=df.index)
#     return standardized_df

# # 신호 처리 기법 선택
# processing_technique = st.sidebar.selectbox("Signal Processing Technique", ["Raw", "Normalized", "Standardized"])

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
#             processed_df = df
#         elif processing_technique == "Normalized":
#             processed_df = normalize_data(df)
#         elif processing_technique == "Standardized":
#             processed_df = standardize_data(df)

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










# import streamlit as st
# import numpy as np
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import time

# # 데이터 가져오기
# def get_realtime_data(file_name):
#     base_url = "https://github.com/PAK917/TimingBelt/tree/main/CMPS_20230531"
#     url = base_url + file_name
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # 오류가 발생하면 예외 발생
#         data = response.text
#         return data
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error retrieving data: {str(e)}")
#         return None

# # 정규화 함수
# def normalize_data(df):
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(df)
#     scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
#     return scaled_df

# # 표준화 함수
# def standardize_data(df):
#     scaler = StandardScaler()
#     standardized_data = scaler.fit_transform(df)
#     standardized_df = pd.DataFrame(standardized_data, columns=df.columns, index=df.index)
#     return standardized_df

# # 신호 처리 기법 선택
# processing_technique = st.sidebar.selectbox("Signal Processing Technique", ["Raw", "Normalized", "Standardized"])

# # 스트림릿 앱 시작
# st.header("Real-time Data Visualization")

# # 버튼 클릭 여부 확인
# button_clicked = st.sidebar.button("Show Graph")

# # 데이터 처리 및 시각화
# start_time = time.strptime("20230531_205400", "%Y%m%d_%H:%M%S")
# end_time = time.strptime("20230531_211039", "%Y%m%d_%H:%M%S")
# current_time = start_time

# #fig, ax = plt.subplots(figsize=(12, 4))
# df_combined = pd.DataFrame()  # 이전 데이터와 새로운 데이터를 합칠 데이터프레임

# chart = st.line_chart()  # Create an empty line chart

# while current_time <= end_time:
#     #file_name = time.strftime("A%Y-%m-%d %H:%M:%S.txt", current_time)
#     file_name = time.strftime("A%Y-%m-%d %H:%M.csv", current_time)
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
#             processed_df = df
#         elif processing_technique == "Normalized":
#             processed_df = normalize_data(df)
#         elif processing_technique == "Standardized":
#             processed_df = standardize_data(df)

#         # 기존 데이터프레임과 새로운 데이터프레임 합치기
#         df_combined = pd.concat([df_combined, processed_df], ignore_index=True)

#         # 그래프 그리기
#         # ax.clear()
#         # for column in processed_df.columns:
#         #     ax.plot(df_combined.index, df_combined[column], label=column)
#         # ax.set_xlim(0, len(df_combined))
#         # ax.set_ylim(df_combined.min().min(), df_combined.max().max())
#         # ax.set_xlabel("Time")
#         # ax.set_ylabel("Value")
#         # ax.set_title(f"Real-time Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#         # ax.legend()

#         # Update the line chart with new rows
#         chart.add_rows(processed_df)

#         # Display the updated line chart
#         # st.pyplot(fig)

#     # 다음 시간으로 업데이트
#     current_time = time.localtime(time.mktime(current_time) + 60)  # 60 seconds = 1 minute

#     # 1초마다 데이터 갱신
#     time.sleep(1)

# # 그래프 창 닫기
# # plt.close(fig)



####################이 위에가 원래 코드 밑에는 정규꺼로 다시 만들어 본 코드

# import streamlit as st
# import numpy as np
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import time
# import io

# # 데이터 가져오기
# def get_realtime_data(file_name):
#     base_url = "https://raw.githubusercontent.com/PAK917/TimingBelt/main/CMPS_20230531/"
#     url = base_url + file_name
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # 오류가 발생하면 예외 발생
#         data = response.text
#         return data
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error retrieving data: {str(e)}")
#         return None

# # 정규화 함수
# def normalize_data(df):
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(df)
#     scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
#     return scaled_df

# # 표준화 함수
# def standardize_data(df):
#     scaler = StandardScaler()
#     standardized_data = scaler.fit_transform(df)
#     standardized_df = pd.DataFrame(standardized_data, columns=df.columns, index=df.index)
#     return standardized_df

# # 신호 처리 기법 선택
# processing_technique = st.sidebar.selectbox("Signal Processing Technique", ["Raw", "Normalized", "Standardized"])

# # 스트림릿 앱 시작
# st.header("Real-time Data Visualization")

# # 버튼 클릭 여부 확인
# button_clicked = st.sidebar.button("Show Graph")

# # 데이터 처리 및 시각화
# start_time = time.strptime("20230531_205400", "%Y%m%d_%H%M%S")
# end_time = time.strptime("20230531_211039", "%Y%m%d_%H%M%S")
# current_time = start_time

# df_combined = pd.DataFrame()  # 이전 데이터와 새로운 데이터를 합칠 데이터프레임

# chart = st.line_chart()  # Create an empty line chart

# while current_time <= end_time:
#     file_name = time.strftime("%Y%m%d_%H%M%S.csv", current_time)
#     data = get_realtime_data(file_name)
#     if data and button_clicked:
#         # 데이터 전처리
#         df = pd.read_csv(io.StringIO(data), skiprows=1, names=["Number", "Time", "External Sound", "Ambient Temp.", "Target Temp"])
#         df["Time"] = pd.to_datetime(df["Time"])
#         df = df.set_index("Time")

#         # 선택한 신호 처리 기법 적용
#         if processing_technique == "Raw":
#             processed_df = df
#         elif processing_technique == "Normalized":
#             processed_df = normalize_data(df)
#         elif processing_technique == "Standardized":
#             processed_df = standardize_data(df)

#         # 기존 데이터프레임과 새로운 데이터프레임 합치기
#         df_combined = pd.concat([df_combined, processed_df], ignore_index=True)

#         # Update the line chart with new rows
#         chart.add_rows(processed_df)

#     # 다음 시간으로 업데이트
#     current_time = time.localtime(time.mktime(current_time) + 60)  # 60 seconds = 1 minute

#     # 1초마다 데이터 갱신
#     time.sleep(1)

###########번호부분은 안가져오고 시간이랑 센서만 가져오는 코드

import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import io

# 데이터 가져오기
def get_realtime_data(file_name):
    base_url = "https://raw.githubusercontent.com/PAK917/TimingBelt/main/CMPS_20230531/"
    url = base_url + file_name
    try:
        response = requests.get(url)
        response.raise_for_status()  # 오류가 발생하면 예외 발생
        data = response.text
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"데이터를 가져오는 중 오류 발생: {str(e)}")
        return None

# 정규화 함수
def normalize_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    return scaled_df

# 표준화 함수
def standardize_data(df):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df)
    standardized_df = pd.DataFrame(standardized_data, columns=df.columns, index=df.index)
    return standardized_df

# 신호 처리 기법 선택
processing_technique = st.sidebar.selectbox("신호 처리 기법", ["Raw", "정규화", "표준화"])

# 스트림릿 앱 시작
st.header("실시간 데이터 시각화")

# 버튼 클릭 여부 확인
button_clicked = st.sidebar.button("그래프 표시")

# 데이터 처리 및 시각화
start_time = time.strptime("20230531_205400", "%Y%m%d_%H%M%S")
end_time = time.strptime("20230531_211039", "%Y%m%d_%H%M%S")
current_time = start_time

df_combined = pd.DataFrame()  # 이전 데이터와 새로운 데이터를 합칠 데이터프레임

chart = st.line_chart()  # 빈 라인 차트 생성

while current_time <= end_time:
    file_name = time.strftime("%Y%m%d_%H%M%S.csv", current_time)
    data = get_realtime_data(file_name)
    if data and button_clicked:
        # 데이터 전처리
        df = pd.read_csv(io.StringIO(data), skiprows=1, usecols=[1, 2, 3, 4], names=["Time", "외부 소음", "주변 온도", "목표 온도"])
        df["Time"] = pd.to_datetime(df["Time"])
        df = df.set_index("Time")

        # 선택한 신호 처리 기법 적용
        if processing_technique == "Raw":
            processed_df = df
        elif processing_technique == "정규화":
            processed_df = normalize_data(df)
        elif processing_technique == "표준화":
            processed_df = standardize_data(df)

        # 기존 데이터프레임과 새로운 데이터프레임 합치기
        df_combined = pd.concat([df_combined, processed_df], ignore_index=True)

        # 새로운 행으로 라인 차트 업데이트
        chart.add_rows(processed_df)

    # 다음 시간으로 업데이트
    current_time = time.localtime(time.mktime(current_time) + 60)  # 60초 = 1분

    # 1초마다 데이터 갱신
    time.sleep(1)
















#############raw데이터, 정규화 스케일링, 표준화스케일링 선택 가능



############ 측면파손 데이터#############

# import streamlit as st
# import numpy as np
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import time

# # 데이터 가져오기
# def get_realtime_data(file_name):
#     base_url = "https://raw.githubusercontent.com/changyeon99/TimingBeltData_CMPS/main/"
#     url = base_url + file_name
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # 오류가 발생하면 예외 발생
#         data = response.text
#         return data
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error retrieving data: {str(e)}")
#         return None

# # 정규화 함수
# def normalize_data(df):
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(df)
#     scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
#     return scaled_df

# # 표준화 함수
# def standardize_data(df):
#     scaler = StandardScaler()
#     standardized_data = scaler.fit_transform(df)
#     standardized_df = pd.DataFrame(standardized_data, columns=df.columns, index=df.index)
#     return standardized_df

# # 신호 처리 기법 선택
# processing_technique = st.sidebar.selectbox("Signal Processing Technique", ["Raw", "Normalized", "Standardized"])

# # 스트림릿 앱 시작
# st.header("Real-time Data Visualization")

# # 버튼 클릭 여부 확인
# button_clicked = st.sidebar.button("Show Graph")

# # 데이터 처리 및 시각화
# start_time = time.strptime("2023-05-29 19:50", "%Y-%m-%d %H:%M")
# end_time = time.strptime("2023-05-29 22:34", "%Y-%m-%d %H:%M")
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
#             processed_df = df
#         elif processing_technique == "Normalized":
#             processed_df = normalize_data(df)
#         elif processing_technique == "Standardized":
#             processed_df = standardize_data(df)

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
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.text
#         return data
#     else:
#         return None

# # 스트림릿 앱 시작
# st.header("Real-time Data Visualization")

# # 버튼 클릭 상태
# button_clicked = False

# # 버튼 추가
# if st.button("Generate Graph and Data"):
#     button_clicked = True

# # 데이터 처리 및 시각화
# start_time = time.strptime("2023-05-28 00:03", "%Y-%m-%d %H:%M")
# end_time = time.strptime("2023-05-28 10:46", "%Y-%m-%d %H:%M")
# current_time = start_time

# while current_time <= end_time:
#     file_name = time.strftime("A%Y-%m-%d %H:%M.txt", current_time)
#     data = get_realtime_data(file_name)
#     if data:
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

#         if button_clicked:
#             # 그래프 설명
#             st.write(f"Real-time Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#             st.line_chart(df)

#             # 데이터 출력
#             st.write("Data:")
#             st.dataframe(df)

#     # 다음 시간으로 업데이트
#     current_time = time.localtime(time.mktime(current_time) + 60)  # 60 seconds = 1 minute

#     # 1초마다 데이터 갱신
#     time.sleep(1)




