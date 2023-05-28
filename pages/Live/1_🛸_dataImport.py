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


import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import time

# 데이터 가져오기
def get_realtime_data():
    base_url = "https://raw.githubusercontent.com/changyeon99/TimingBeltData/main/"
    current_time = time.strftime("%Y-%m-%d %H:%M")
    file_name = f"A{current_time.replace(' ', '%20').replace(':', '')}.txt"
    url = base_url + file_name
    response = requests.get(url)
    if response.status_code == 200:
        data = response.text
        return data
    else:
        return None

# 스트림릿 앱 시작
st.header("실시간 데이터 시각화")

# 데이터 처리 및 시각화
while True:
    data = get_realtime_data()
    if data:
        # 데이터 전처리
        lines = data.strip().split("\n")
        values = []
        for line in lines:
            if line:
                try:
                    parts = line.split(",")
                    value = float(parts[0])  # 센서값이 첫 번째 열에 위치하는 것으로 가정
                    values.append(value)
                except ValueError:
                    continue

        # 데이터프레임 생성
        df = pd.DataFrame({"Value": values})

        # 그래프 생성
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df["Value"])
        ax.set_xlabel("시간")
        ax.set_ylabel("값")
        ax.set_title("실시간 데이터")
        st.pyplot(fig)

    # 1초마다 데이터 갱신
    time.sleep(1)





# import streamlit as st
# import numpy as np
# import pandas as pd
# import requests
# import plotly.graph_objects as go
# import io
# import time

# # 데이터 가져오기
# def get_realtime_data():
#     url = "https://raw.githubusercontent.com/changyeon99/TimingBeltData/main/"
#     response = requests.get(url)
#     if response.status_code == 200:
#         content = response.text
#         filenames = content.strip().split("\n")
#         data = []
#         for filename in filenames:
#             file_url = url + filename
#             file_response = requests.get(file_url)
#             if file_response.status_code == 200:
#                 file_content = file_response.text
#                 df = pd.read_csv(io.StringIO(file_content), sep='\t', header=None)
#                 data.append(df)
#         return data
#     else:
#         return None

# # 스트림릿 앱 시작
# st.header("Real-time Data Visualization")

# # 시작 버튼
# start_button = st.button("Start")

# # 그래프 업데이트 함수
# def update_graph():
#     fig = go.Figure()
#     graph_placeholder = st.plotly_chart(fig)

#     while True:
#         data = get_realtime_data()
#         if data:
#             for df in data:
#                 # 데이터 처리
#                 x = df.index
#                 y = df.iloc[:, 0:3]

#                 # 그래프 생성
#                 fig = go.Figure()
#                 for column in y.columns:
#                     fig.add_trace(go.Scatter(x=x, y=y[column], name=column))
#                 fig.update_layout(title="Data Visualization")

#                 # 그래프 업데이트
#                 graph_placeholder.plotly_chart(fig)

#                 # 1초마다 데이터 갱신
#                 time.sleep(1)

# # 데이터 처리 및 시각화
# if start_button:
#     update_graph()



