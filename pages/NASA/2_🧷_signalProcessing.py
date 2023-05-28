import streamlit as st
import numpy as np
import pandas as pd
import glob #파일 다 읽어줌
import os, sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')  # 또는 'Qt5Agg' 또는 시스템과 호환되는 다른 백엔드


os.chdir("C:\\Users\\San\\Desktop\\23-1\\cap\\practice\\샘플 데이터\\NASA Bearing Dataset\\2nd_test")
# os.chdir=change directory로 작업중인 디렉토리를 변경하는 함수
file_list = glob.glob("**/*.39", recursive=True)
#"**/*.39": 이 패턴은 현재 디렉토리 및 모든 하위 디렉토리에서 확장자가 .39인 파일을 찾습니다.
#"**/"는 현재 디렉토리 및 모든 하위 디렉토리를 의미합니다.
#"*.39"는 확장자가 .39인 파일을 의미합니다.
#recursive=True: 이 옵션은 모든 하위 디렉토리에서도 파일을 검색하도록 합니다.
#glove함수는 읽어주는 함수

#len(file_list) # 행의 길이(2번째 테스트 데이터)

df = pd.read_csv(file_list[0], sep='\t',header=None) #file_list[0]에 있는걸 읽어오는 행 파일이 tab으로 구분되어있음. 


#rms 신호추출기법 root mean square 신호의 크기를 추출
def rms(stats):  #rms를 이용하여 신호추출(제곱 한 값의 평균의 루트를 씌우는 것)
    return(np.sqrt(np.mean(stats**2, axis=0)))# 열방향으로 stats을 제곱한것에대한 평균의 제곱근 반환 그래서 결과값이 베어링 갯수만큼 4개
rms(df)
# 2000개의 데이터 샘플을 하나의 파일형 데이터 파일로 추출 
# 파일이 984개였는데 추출을 하면 row(행)가 984개인 파일로 추출될 것이다

#984개 파일을 하나의 csv로 만드는 과정
rms_arr = np.array([])     #비어있는 배열을 만든다
for file in file_list:     #위에 정의한 file_list에 있는 파일을 하나씩 가져온다.
    # 데이터파일 하나씩 읽기(984개를 다 읽어야해서 반복문 활용)
    df = pd.read_csv(file, sep='\t',header=None)
    # rms value 추출
    rms_value = rms(df.values)
    
    rms_arr = np.concatenate([rms_arr, rms_value], axis=0)#이전에 만든 rms_arr에다 rms_value를 이어붙이는 코드(추출한 rms를 빈 배열에 넣는 코드)
rms_arr = rms_arr.reshape(len(file_list), 4)#rms_arr을 다차원 배열로 변환한다.

rms_df = pd.DataFrame(rms_arr, columns=['ch1','ch2','ch3','ch4'])#rms_arr을 데이터 프레임으로 생성한다.(열=변수(여기선 베어링 4개))
rms_df.to_csv('C:\\Users\\San\\Desktop\\23-1\\cap\\practice\\샘플 데이터\\NASA Bearing Dataset\\RMS_bearing.csv', index=None) # 저장하는 코드(원하는 경로, 원하는파일 이름 여기선 RMS_bearing.csv)
# rms_df = None

# 스트림릿 앱의 헤더 및 설명 추가
st.header("RMS 데이터 시각화")
st.write("RMS 데이터를 플로팅합니다.")

# Matplotlib figure 생성
with st.spinner('로딩 중...'):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rms_df)

# 플롯을 스트림릿 앱에 표시
st.pyplot(fig)






#plt.figure(figsize=(12,4))
#plt.plot(rms_df.values) #rms_df.values=rms_df내의 모든 값들을 Numpy배열 형태로 가져오는 것.

#이 값을 사용하여 NumPy 배열로 RMS 데이터를 처리하거나, 다른 NumPy 배
#열과 결합하거나, NumPy 배열의 수학 및 통계 함수를 적용하거나, 시각화를 

#위한 플로팅 라이브러리와 함께 사용할 수 있습니다.
#plt.show()

# 실 수집 데이터에서 이쁘게 나오지 않으면 rms가 아닌 다른 기법을 다시 적용해야함
#신호추출이 잘 되었다.(실제로는 다른 신호처리를 다시써야할수도..(표준편차,평균 등 예쁘게 나오게 적용해봐야 한다.))
