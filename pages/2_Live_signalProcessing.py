# import streamlit as st
# import numpy as np
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt
# from scipy.fft import fft
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

# # RMS 신호처리 함수
# def rms(stats):
#     return np.sqrt(np.mean(stats**2, axis=0))

# # FFT 신호처리 함수
# def fft_signal(stats):
#     return np.abs(fft(stats, axis=0))

# # 스트림릿 앱 시작
# st.header("Real-time Data Visualization")
# st.subheader("Select Signal Processing Technique")

# # 신호처리 기법 선택
# processing_technique = st.selectbox("Processing Technique", ["RMS", "FFT"])

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

#         # 선택된 신호처리 기법 적용
#         if processing_technique == "RMS":
#             processed_values = rms(df.values)
#             processed_df = pd.DataFrame(processed_values, columns=["RMS Value"])

#             # 그래프 생성
#             fig, ax = plt.subplots(figsize=(12, 4))
#             ax.plot(processed_df.index, processed_df["RMS Value"])
#             ax.set_xlabel("Time")
#             ax.set_ylabel("RMS Value")
#             ax.set_title(f"RMS Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#             st.pyplot(fig)

#             # 데이터 출력
#             st.dataframe(processed_df)

#         elif processing_technique == "FFT":
#             processed_values = fft_signal(df.values)
#             processed_df = pd.DataFrame(processed_values, columns=["FFT Value"])

#             # 그래프 생성
#             fig, ax = plt.subplots(figsize=(12, 8))
#             for i in range(processed_df.shape[1]):
#                 ax.plot(processed_df.index, processed_df.iloc[:, i], label=f"Ch{i+1} FFT")
#             ax.plot(processed_df.values.flatten(), label="Combined FFT", color="black", linewidth=2)
#             ax.legend()
#             ax.set_xlabel("Frequency")
#             ax.set_ylabel("Amplitude")
#             ax.set_title(f"FFT Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#             st.pyplot(fig)

#             # 데이터 출력
#             st.dataframe(processed_df)

#     # 다음 시간으로 업데이트
#     current_time = time.localtime(time.mktime(current_time) + 60)  # 60 seconds = 1 minute

#     # 1초마다 데이터 갱신
#     time.sleep(1)




# import streamlit as st
# import numpy as np
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt
# from scipy.fft import fft
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

# # RMS 신호처리 함수
# def rms(stats):
#     return np.sqrt(np.mean(stats**2, axis=0))

# # FFT 신호처리 함수
# def fft_signal(stats):
#     return np.abs(fft(stats, axis=0))

# # 스트림릿 앱 시작
# st.header("Real-time Data Visualization")
# st.subheader("Select Signal Processing Technique")

# # 신호처리 기법 선택
# processing_technique = st.selectbox("Processing Technique", ["RMS", "FFT"])

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
#             parts = line.strip().split(",")
#             if len(parts) >= 3:
#                 if parts[0] != '' and parts[1] != '' and parts[2] != '':
#                     try:
#                         value_1 = float(parts[0])
#                         value_2 = float(parts[1])
#                         value_3 = float(parts[2])
#                         values_1.append(value_1)
#                         values_2.append(value_2)
#                         values_3.append(value_3)
#                     except ValueError:
#                         # 부동 소수점으로 변환할 수 없는 문자열이 있는 경우 건너뜁니다.
#                         continue

#         df = pd.DataFrame({"ch1": values_1, "ch2": values_2, "ch3": values_3})

#     # 신호처리 기법 선택에 따른 처리
#     if processing_technique == "RMS":
#         processed_values = rms(df.values)
#         processed_df = pd.DataFrame(processed_values.reshape(1, -1), columns=[f"RMS Value {i+1}" for i in range(processed_values.shape[0])])

#         # 그래프 출력
#         fig, ax = plt.subplots(figsize=(12, 4))
#         ax.plot(processed_df.values.flatten())  # 수정된 부분
#         #ax.plot(processed_values)
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Value")
#         ax.set_title(f"{processing_technique} Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#         st.pyplot(fig)

#     else:
#         processed_values = fft_signal(df.values)
#         processed_df = pd.DataFrame(processed_values, columns=[f"FFT Value {i+1}" for i in range(processed_values.shape[1])])

#         # 그래프 출력
#         fig, ax = plt.subplots(figsize=(12, 4))
#         for column in processed_df.columns:
#             ax.plot(processed_df[column])
#         ax.legend(processed_df.columns)
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Value")
#         ax.set_title(f"{processing_technique} Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#         st.pyplot(fig)


#         # 데이터 출력
#         st.dataframe(processed_df)

#     # 다음 시간으로 업데이트
#     current_time = time.localtime(time.mktime(current_time) + 60)  # 60 seconds = 1 minute

#     # 그래프 닫기
#     plt.close()

#     # 1초마다 데이터 갱신
#     time.sleep(1)

# import streamlit as st
# import numpy as np
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt
# from scipy.fft import fft
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

# # RMS 신호처리 함수
# def rms(stats):
#     return np.sqrt(np.mean(stats ** 2, axis=0))

# # FFT 신호처리 함수
# def fft_signal(stats):
#     return np.abs(fft(stats, axis=0))

# # 스트림릿 앱 시작
# st.header("Real-time Data Visualization")
# st.subheader("Select Signal Processing Technique")

# # 신호처리 기법 선택
# processing_technique = st.selectbox("Processing Technique", ["RMS", "FFT"])

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
#             parts = line.strip().split(",")
#             if len(parts) >= 3:
#                 if parts[0] != '' and parts[1] != '' and parts[2] != '':
#                     try:
#                         value_1 = float(parts[0])
#                         value_2 = float(parts[1])
#                         value_3 = float(parts[2])
#                         values_1.append(value_1)
#                         values_2.append(value_2)
#                         values_3.append(value_3)
#                     except ValueError:
#                         # 부동 소수점으로 변환할 수 없는 문자열이 있는 경우 건너뜁니다.
#                         continue

#         df = pd.DataFrame({"ch1": values_1, "ch2": values_2, "ch3": values_3})

#         # 신호처리 기법 선택에 따른 처리
#         if processing_technique == "RMS":
#             processed_values = rms(df.values)
#             processed_df = pd.DataFrame(processed_values.reshape(-1, 1), columns=["RMS Values"])

#             # 그래프 출력
#             fig, ax = plt.subplots(figsize=(12, 4))
#             for column in processed_df.columns:
#                 ax.plot(processed_df[column])
#             ax.legend(processed_df.columns)
#             ax.set_xlabel("Time")
#             ax.set_ylabel("Value")
#             ax.set_title(f"{processing_technique} Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#             st.pyplot(fig)

#         else:
#             processed_values = fft_signal(df.values)
#             processed_df = pd.DataFrame(processed_values, columns=[f"FFT Value {i + 1}" for i in range(processed_values.shape[1])])

#             # 그래프 출력
#             fig, ax = plt.subplots(figsize=(12, 4))
#             for column in processed_df.columns:
#                 ax.plot(processed_df[column])
#             ax.legend(processed_df.columns)
#             ax.set_xlabel("Time")
#             ax.set_ylabel("Value")
#             ax.set_title(f"{processing_technique} Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#             st.pyplot(fig)

#             # 데이터 출력
#             st.dataframe(processed_df)

#     # 다음 시간으로 업데이트
#     current_time = time.localtime(time.mktime(current_time) + 60)  # 60 seconds = 1 minute

#     # 1초마다 데이터 갱신
#     time.sleep(1)


## 여기서 rms기법은 끝이 없이 실시간으로 데이터가 들어오기때문에 기존 방법대로 사용할 수가 없다
## 왜냐하면 파일안에있는 1~3센서값을 1번파일부터 마지막 파일까지의 센서값들을 제곱해서 평균을 내고 루트를 씌우는 것
## 그렇기 때문에 끝없이 데이터를 받게 된다면 데이터를 받을 때마다 계산해야될 것이다. 즉, 끝이 있는 데이터를 사용할때 
## 사용해야한다. 그러므로 여기서의 rms는 한 파일에 있는 모든 센서에대해 계산된 값 각각의 센서에대해 계산된 값이 아니다.


# import streamlit as st
# import numpy as np
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt
# from scipy.fft import fft
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

# # RMS 신호처리 함수
# def rms(stats):
#     return np.sqrt(np.mean(stats ** 2, axis=0))

# # FFT 신호처리 함수
# def fft_signal(stats):
#     return np.abs(fft(stats, axis=0))

# # 스트림릿 앱 시작
# st.header("실시간 데이터 시각화")
# st.subheader("신호 처리 기법 선택")

# # 신호 처리 기법 선택
# processing_technique = st.selectbox("신호 처리 기법", ["RMS", "FFT"])

# # 그래프 카운터
# graph_counter = 0

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
#             parts = line.strip().split(",")
#             if len(parts) >= 3:
#                 if parts[0] != '' and parts[1] != '' and parts[2] != '':
#                     try:
#                         value_1 = float(parts[0])
#                         value_2 = float(parts[1])
#                         value_3 = float(parts[2])
#                         values_1.append(value_1)
#                         values_2.append(value_2)
#                         values_3.append(value_3)
#                     except ValueError:
#                         # 부동 소수점으로 변환할 수 없는 문자열이 있는 경우 건너뜁니다.
#                         continue

#         df = pd.DataFrame({"ch1": values_1, "ch2": values_2, "ch3": values_3})

#         # 신호 처리 기법 선택에 따른 처리
#         if processing_technique == "RMS":
#             processed_values = rms(df.values)
#             processed_df = pd.DataFrame(processed_values.reshape(-1, 1), columns=["RMS value"])

#             # 그래프 출력
#             if graph_counter >= 10:
#                 plt.close('all')  # 이전 그래프 모두 닫기
#                 graph_counter = 0

#             fig, ax = plt.subplots(figsize=(12, 4))
#             for column in processed_df.columns:
#                 ax.plot(processed_df[column])
#             ax.legend(processed_df.columns)
#             ax.set_xlabel("Time")
#             ax.set_ylabel("Value")
#             ax.set_title(f"{processing_technique} Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#             st.pyplot(fig)
#             graph_counter += 1

#         else:
#             processed_values = fft_signal(df.values)
#             processed_df = pd.DataFrame(processed_values, columns=[f"FFT value {i + 1}" for i in range(processed_values.shape[1])])

#             # 그래프 출력
#             if graph_counter >= 10:
#                 plt.close('all')  # 이전 그래프 모두 닫기
#                 graph_counter = 0

#             fig, ax = plt.subplots(figsize=(12, 4))
#             for column in processed_df.columns:
#                 ax.plot(processed_df[column])
#             ax.legend(processed_df.columns)
#             ax.set_xlabel("Time")
#             ax.set_ylabel("Value")
#             ax.set_title(f"{processing_technique} Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#             st.pyplot(fig)
#             graph_counter += 1

#             # 데이터 출력
#             st.dataframe(processed_df)

#     # 다음 시간으로 업데이트
#     current_time = time.localtime(time.mktime(current_time) + 60)  # 60초 = 1분

#     # 1초마다 데이터 갱신
#     time.sleep(1)


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

# B 코드의 rms 신호처리 함수
def rms(stats):
    return np.sqrt(np.mean(stats ** 2, axis=0))

# FFT 신호처리 함수
def fft_signal(stats):
    return np.abs(fft(stats, axis=0))

# 스트림릿 앱 시작
st.header("실시간 데이터 시각화")
st.subheader("신호 처리 기법 선택")

# 신호 처리 기법 선택
processing_technique = st.selectbox("신호 처리 기법", ["RMS", "FFT"])

# 그래프 카운터
graph_counter = 0

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

        # 신호 처리 기법 선택에 따른 처리
        if processing_technique == "RMS":
            processed_values = rms(df.values)
            processed_df = pd.DataFrame(processed_values.reshape(-1, 1), columns=["RMS value"])

            # 그래프 출력
            if graph_counter >= 10:
                plt.close('all')  # 이전 그래프 모두 닫기
                graph_counter = 0

            fig, ax = plt.subplots(figsize=(12, 4))
            for column in processed_df.columns:
                ax.plot(processed_df[column])
            ax.legend(processed_df.columns)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.set_title(f"{processing_technique} Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
            st.pyplot(fig)
            graph_counter += 1

        else:
            processed_values = fft_signal(df.values)
            processed_df = pd.DataFrame(processed_values, columns=[f"FFT value {i + 1}" for i in range(processed_values.shape[1])])

            # 그래프 출력
            if graph_counter >= 10:
                plt.close('all')  # 이전 그래프 모두 닫기
                graph_counter = 0

            fig, ax = plt.subplots(figsize=(12, 4))
            for column in processed_df.columns:
                ax.plot(processed_df[column])
            ax.legend(processed_df.columns)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.set_title(f"{processing_technique} Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
            st.pyplot(fig)
            graph_counter += 1

            # 데이터 출력
            st.dataframe(processed_df)

    # 다음 시간으로 업데이트
    current_time = time.localtime(time.mktime(current_time) + 60)  # 60초 = 1분

    # 1초마다 데이터 갱신
    time.sleep(1)

## 여기서 rms기법은 끝이 없이 실시간으로 데이터가 들어오기때문에 기존 방법대로 사용할 수가 없다
## 왜냐하면 파일안에있는 1~3센서값을 1번파일부터 마지막 파일까지의 센서값들을 제곱해서 평균을 내고 루트를 씌우는 것
## 그렇기 때문에 끝없이 데이터를 받게 된다면 데이터를 받을 때마다 계산해야될 것이다. 즉, 끝이 있는 데이터를 사용할때 
## 사용해야한다. 그러므로 여기서의 rms는 한 파일에 있는 모든 센서에대해 계산된 값 각각의 센서에대해 계산된 값이 아니다.









################## RMS 값을 각 센서의 RMS값으로 추출하여 그래프 만드는 코드


# import streamlit as st
# import numpy as np
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt
# from scipy.fft import fft
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

# # B 코드의 rms 신호처리 함수
# def rms(stats):
#     return np.sqrt(np.mean(stats ** 2, axis=0))

# # FFT 신호처리 함수
# def fft_signal(stats):
#     return np.abs(fft(stats, axis=0))

# # 스트림릿 앱 시작
# st.header("실시간 데이터 시각화")
# st.subheader("신호 처리 기법 선택")

# # 신호 처리 기법 선택
# processing_technique = st.selectbox("신호 처리 기법", ["RMS", "FFT"])

# # 그래프 카운터
# graph_counter = 0

# # 데이터 처리 및 시각화
# start_time = time.strptime("2023-05-28 00:03", "%Y-%m-%d %H:%M")
# end_time = time.strptime("2023-05-28 10:46", "%Y-%m-%d %H:%M")
# current_time = start_time

# rms_values = []

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
#             parts = line.strip().split(",")
#             if len(parts) >= 3:
#                 if parts[0] != '' and parts[1] != '' and parts[2] != '':
#                     try:
#                         value_1 = float(parts[0])
#                         value_2 = float(parts[1])
#                         value_3 = float(parts[2])
#                         values_1.append(value_1)
#                         values_2.append(value_2)
#                         values_3.append(value_3)
#                     except ValueError:
#                         # 부동 소수점으로 변환할 수 없는 문자열이 있는 경우 건너뜁니다.
#                         continue

#         df = pd.DataFrame({"ch1": values_1, "ch2": values_2, "ch3": values_3})

#         # 신호 처리 기법 선택에 따른 처리
#         if processing_technique == "RMS":
#             processed_values = rms(df.values)
#             rms_values.append(processed_values)

#             if len(rms_values) > 10:
#                 rms_values = rms_values[1:]  # 최근 10개 값만 유지

#             # 그래프 출력
#             if graph_counter >= 10:
#                 plt.close('all')  # 이전 그래프 모두 닫기
#                 graph_counter = 0

#             fig, ax = plt.subplots(figsize=(12, 4))
#             for i in range(3):
#                 ax.plot(np.arange(len(rms_values)), [value[i] for value in rms_values], label=f"Sensor {i+1}")
#             ax.set_xlabel("Time")
#             ax.set_ylabel("RMS Value")
#             ax.set_title(f"RMS Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#             ax.legend()
#             st.pyplot(fig)
#             graph_counter += 1

#         else:
#             processed_values = fft_signal(df.values)
#             processed_df = pd.DataFrame(processed_values, columns=[f"FFT value {i + 1}" for i in range(processed_values.shape[1])])

#             # 그래프 출력
#             if graph_counter >= 10:
#                 plt.close('all')  # 이전 그래프 모두 닫기
#                 graph_counter = 0

#             fig, ax = plt.subplots(figsize=(12, 4))
#             for column in processed_df.columns:
#                 ax.plot(processed_df[column])
#             ax.legend(processed_df.columns)
#             ax.set_xlabel("Time")
#             ax.set_ylabel("Value")
#             ax.set_title(f"{processing_technique} Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
#             st.pyplot(fig)
#             graph_counter += 1

#             # 데이터 출력
#             st.dataframe(processed_df)

#     # 다음 시간으로 업데이트
#     current_time = time.localtime(time.mktime(current_time) + 60)  # 60초 = 1분

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
#     return df

# # 신호 처리 함수 - RMS
# def process_rms_data(df):
#     processed_df = pd.DataFrame(index=df.index)  # 인덱스 설정
#     for column in df.columns:
#         processed_column = df[column].dropna()  # NaN 값 제거
#         processed_df[column] = np.sqrt(np.mean(np.square(processed_column)))
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











#plt.figure(figsize=(12,4))
#plt.plot(rms_df.values) #rms_df.values=rms_df내의 모든 값들을 Numpy배열 형태로 가져오는 것.

#이 값을 사용하여 NumPy 배열로 RMS 데이터를 처리하거나, 다른 NumPy 배
#열과 결합하거나, NumPy 배열의 수학 및 통계 함수를 적용하거나, 시각화를 

#위한 플로팅 라이브러리와 함께 사용할 수 있습니다.
#plt.show()

# 실 수집 데이터에서 이쁘게 나오지 않으면 rms가 아닌 다른 기법을 다시 적용해야함
#신호추출이 잘 되었다.(실제로는 다른 신호처리를 다시써야할수도..(표준편차,평균 등 예쁘게 나오게 적용해봐야 한다.))
