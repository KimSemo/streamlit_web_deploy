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
    response = requests.get(url)
    if response.status_code == 200:
        data = response.text
        return data
    else:
        return None

# 스트림릿 앱 시작
st.header("Real-time Data Visualization")

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

        # 그래프 생성
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df["Value 1"], label="Value 1")
        ax.plot(df.index, df["Value 2"], label="Value 2")
        ax.plot(df.index, df["Value 3"], label="Value 3")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title(f"Real-time Data - {time.strftime('%Y-%m-%d %H:%M', current_time)}")
        ax.legend()
        st.pyplot(fig)

        # 그래프 삭제
        plt.close(fig)

        # 데이터 출력
        st.dataframe(df)

    # 다음 시간으로 업데이트
    current_time = time.localtime(time.mktime(current_time) + 60)  # 60 seconds = 1 minute

    # 1초마다 데이터 갱신
    time.sleep(1)

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




