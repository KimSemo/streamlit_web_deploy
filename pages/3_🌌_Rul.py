import numpy as np
import pandas as pd
import glob #파일 다 읽어줌
import os, sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

os.chdir("C:\\Users\\San\\Desktop\\23-1\\cap\\practice\\샘플 데이터\\NASA Bearing Dataset\\2nd_test")
# os.chdir=change directory로 작업중인 디렉토리를 변경하는 함수
file_list = glob.glob("**/*.39", recursive=True)
#"**/*.39": 이 패턴은 현재 디렉토리 및 모든 하위 디렉토리에서 확장자가 .39인 파일을 찾습니다.
#"**/"는 현재 디렉토리 및 모든 하위 디렉토리를 의미합니다.
#"*.39"는 확장자가 .39인 파일을 의미합니다.
#recursive=True: 이 옵션은 모든 하위 디렉토리에서도 파일을 검색하도록 합니다.

len(file_list) # 행의 길이(2번째 테스트 데이터)

df = pd.read_csv(file_list[0], sep='\t',header=None) #file_list[0]에 있는걸 읽어오는 행 파일이 tab으로 구분되어있음. 
# 첫번째 파일을 불러오는 것
#df.head() # head로 앞에서부터 5개의 값을 뽑음
df
#이렇게 뽑으면 너무 많은(20479*948)데이터가 생기므로 신호처리를 해줘야 한다

plt.figure(figsize=(12,4)) # 사이즈 지정
plt.plot(df)
plt.show()

#rms 신호추출기법 root mean square
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
rms_df.to_csv('C:\\Users\\San\\Desktop\\23-1\\cap\\practice\\샘플 데이터\\NASA Bearing Dataset\\RMS_bearing.csv', index=None) # 저장하는 코드
# rms_df = None

plt.figure(figsize=(12,4))
plt.plot(rms_df.values) #rms_df.values=rms_df내의 모든 값들을 Numpy배열 형태로 가져오는 것.
#이 값을 사용하여 NumPy 배열로 RMS 데이터를 처리하거나, 다른 NumPy 배
#열과 결합하거나, NumPy 배열의 수학 및 통계 함수를 적용하거나, 시각화를 
#위한 플로팅 라이브러리와 함께 사용할 수 있습니다.
plt.show()
# 실 수집 데이터에서 이쁘게 나오지 않으면 rms가 아닌 다른 기법을 다시 적용해야함
#신호추출이 잘 되었다.(실제로는 다른 신호처리를 다시써야할수도..(표준편차,평균 등 예쁘게 나오게 적용해봐야 한다.))

#이상감지의 종류=다양 -MSET=모델링기반(=잔차기반) 이상감지
#여기선 변수가 4개 있으니까 1번 변수를 예측할 때는 2,3,4변수를 사용하고 2번 변수 예측시는 1,3,4변수를 사용
#총 4개의 예측값이 나온다.
#이상감지시 정상을 기준으로 얼마나 떨어져있나 판단-즉 tr데이터는 정상이어야한다.(one class classification)
rms_df = pd.read_csv('C:\\Users\\San\\Desktop\\23-1\\cap\\practice\\샘플 데이터\\NASA Bearing Dataset\\RMS_bearing.csv')
rms_df.head()

# Train Data, Test Data
trdat = rms_df.values[0:400,:]  #400까지만 한다고 친다
tsdat = rms_df.values # 전체데이터를 테스트 데이터로 쓰는건가???->왜냐하면 시계열 데이터여서 연속성을 봐야하기 때문이다!!

trScore_arr = np.zeros([trdat.shape[0], trdat.shape[1]]) #trdata와 같은 크기의 2차원 배열에 내용물은 전부0인 trScore_arr을 만든 것
tsScore_arr = np.zeros([tsdat.shape[0], trdat.shape[1]])

lr = LinearRegression()

input_idx = np.arange(trdat.shape[1]).tolist()
#trdat.shape[1]을 사용하여 trdat의 열크기를 가져온다
#np.arange()를 통하여 0부터 trdat의 열 크기-1까지의 정수를 담은 1차원 배열을 생성한다.
#tolist() 메소드를 사용하여 생성된 1차원 배열을 리스트로 변환
#input_index변수에 trdat배열의 모든 열의 인덱스가 리스트 형태로 저장된다.

#np.arange()의 예시 np.arange(0,10,2) -> [0,2,4,6,8]
for idx in input_idx:
    input_idx = np.arange(trdat.shape[1]).tolist()
    input_idx.remove(idx) # input_idx리스트에서 현재 idx값을 제거합니다.->왜 필요한가?ex.1번 베어링을 예측할때 2,3,4번 베어링을 훈련데이터로 쓰기 때문에 필요하다
    
    lr.fit(trdat[:,input_idx], trdat[:,idx])
    # lr.fit() 사용하여, 학습 데이터의 입력 데이터에서 선택된 변수
    # ('input_idx'리스트에서 제거된 변수들을 제외한 나머지 변수)를 사용하여
    #선형 회귀 모델을 학습시킵니다.
    #trdat[:,input_idx]는 trdat배열에서 input_idx 리스트에 해당하는 열들만 선택한 2차원 부분 배열
    
    
    
    # 각 변수 별 train/test score
    trScore = lr.predict(trdat[:,input_idx])
    tsScore = lr.predict(tsdat[:,input_idx])
    
    trScore_arr[:,idx] = trScore
    tsScore_arr[:,idx] = tsScore

# 각 변수 별 통합 train/test score
integrated_trScore = np.sqrt(np.sum(trScore_arr**2, axis=1))
integrated_tsScore = np.sqrt(np.sum(tsScore_arr**2, axis=1))

#for문부터 코드 끝부분까지의 해석
#전체 변수들의 인덱스를 np.arange(trdat.shape[1])을 사용하여 생성하고, tolist() 메소드를 사용하여 리스트로 변환합니다.
#input_idx 리스트에서 현재 idx 값을 제거합니다.
#lr.fit() 함수를 사용하여, 학습 데이터의 입력 데이터에서 선택된 변수(input_idx 리스트에서 제거된 변수들을 제외한 나머지 변수)를 사용하여 선형 회귀 모델을 학습시킵니다.->ex.1번베어링 훈련시=2,3,4만 훈련
#lr.predict() 함수를 사용하여, 학습 데이터와 테스트 데이터에서 선택된 변수를 사용하여 예측 값을 계산합니다.
#각 변수(idx) 별로 계산된 train/test score 값을, trScore_arr와 tsScore_arr 배열에 저장합니다.
#모든 변수에 대한 train/test score 값(trScore_arr와 tsScore_arr 배열)을 기반으로, 통합 train/test score 값을 계산하여 integrated_trScore와 integrated_tsScore 배열에 저장합니다. 이 때, np.sqrt() 함수를 사용하여 각 변수에 대한 score 값을 모두 더한 후 루트 값을 계산합니다.
#이러한 과정을 통해, 입력 데이터의 변수를 선택하여 선형 회귀 모델을 학습시키고, 변수 선택에 따른 train/test score 값을 계산할 수 있습니다.

plt.figure(figsize=(12,4))
plt.plot(tsScore_arr)
plt.show()

plt.figure(figsize=(12,4))
plt.plot(integrated_tsScore)
plt.show()

# 정상/이상에 대한 기준
# 단측검정
#컨트롤리미트 그리기
def bootlimit(stat, bootstrap, alpha):    
    alpha = 100 - alpha*100
    samsize = len(stat)
    sampled_limit = []
    for i in range(bootstrap):
        sampled_limit.append(np.percentile(np.random.choice(stat, samsize, replace=True), alpha))
    limit = np.mean(sampled_limit)
    return(limit)

cl = bootlimit(integrated_trScore, bootstrap=100, alpha=0.05)

outidx = np.where(integrated_tsScore>cl)[0]

plt.figure(figsize=(12,4))
plt.plot(integrated_tsScore, color='blue')
# plt.plot(outidx, np.repeat(max(integrated_tsScore)*1.1, len(outidx)), 'x', color='red', alpha=0.5)
plt.axhline(y=cl, color='red')

# for idx in outidx:
  #   plt.axvline(x=idx, color='red', linestyle='-', alpha=0.1)



# 이상감지 잔차 값 넣은 컨트롤리미트
plt.show()

#고장날때쯤생기는 진동(큰분산) 완화를 위해 스트레스 누적시킴(ppt참고) l2놈 적용결과 (l2놈=mset) 
#l2Norm을 적용시키면 여러개의 그래프를 하나로 볼 수 있다
l2norm_trScore = np.sqrt(np.sum(trScore_arr**2, axis=1))
#axis=1은 행방향을 따라서 계산하라는 것을 의미

tr_degScore = np.cumsum(l2norm_trScore) / np.arange(1,401,1)
#l2norm_trScore 배열의 누적 합을 구하고, 그것을 해당 인덱스에 따른 평균값으로 나누어 tr_degScore 배열에 저장합니다.
#np.cumsum(l2norm_trScore) 함수는 l2norm_trScore 배열의 누적합을 계산하여 새로운 배열로 반환합니다.
# np.arange(1, 401, 1)은 1부터 400까지 1씩 증가하는 숫자 배열을 생성한 후, tr_degScore의 각 요소와 나누어 주어 해당 인덱스까지의 누적평균 값을 구합니다.



l2norm_tsScore = np.sqrt(np.sum(tsScore_arr**2, axis=1))
ts_degScore = np.cumsum(l2norm_tsScore) / np.arange(1,985,1)

cl = bootlimit(tr_degScore, alpha=0.05, bootstrap=100)
#여기서 alpha는 신뢰수준의 alpha로 95%의 신뢰수준을 계산하도록 지정한 것
#bootstrap 값이 높을수록 부트스트랩 방법을 이용하여 계산한 신뢰 구간이 더 정확하게 계산됩니다.
#디그라데이션결과를 넣어 만든 컨트롤리밑
#정상이상을 측정하는 것 근데, 정상만을 이용해서 기준(cl=임곗값)을 만든다
#정상을 기준으로 관축치가 떨어져있는지를 보는 거니까 정상데이터만을 이용해서 임곗값을 만든다
plt.figure(figsize=(12,4))
plt.plot(ts_degScore, color='blue')
plt.axhline(y=cl, color='red')
plt.show()

# exponential weight 즉, 가중치
def exponential_weight(length, alpha):
    exp_weight = []
    for i in range(1, length+1):
        #길이=984
        w = alpha * (1-alpha)**i
        #ppt에 나온식대로 함수를 구현 한 것 시점이 고장시점에서 멀어질 수록 낮은 값, 가까워질 수록 높은 값을 갖도록 가중치 설정
        
        exp_weight.append(w)
    exp_weight = sorted(exp_weight, reverse=False)
    return(exp_weight)

# Exponentially Weighted Linear Regression
#수명이나 시간 예측 (수명과 디그라데이션 팩터가 있어야한다. x=시간 y=디그라데이션 팩터)
t = 900
#900시점까지만 데이터를 사용하겠다
w = exponential_weight(length=t, alpha=0.99)
#alpha가 클 수록 최근시점 많이 반영 적을 수록 과거시점 많이 반영

# x=시간 y=디그라데이션 팩터
x = np.arange(t)
x = x.reshape(t,1)
#열을 만들어 준 것
y = ts_degScore[0:t].reshape(t,1)

EWLR = LinearRegression(fit_intercept=True)
# intercept=모델의 절편값을 학습할지 물어보는 것 True니까 학습을 한다는 것이다.
EWLR.fit(x,y, sample_weight = w)
#x(시간),y(디그라데이션 팩터),가중치를 학습하는 것

coef = EWLR.coef_
#모델의계수 출력(베타)(y=ax+b에서의 a)
intercept = EWLR.intercept_
#모델의 절편 출력(y=ax+b에서의 b)

#alpha값을 변화시키면서 나타나는 변화가 궁금 녹음에서는 하는것같이 나왔는데 코드에 없어서 어떻게 이루어지는지 보고싶다.

print(coef)
print(intercept)
#디그라에디션 모델구함. y절편 구함 베타 구함

# predefined failure threshold
failure_threshold = max(ts_degScore)
#고장기준(현실적으로 설정하기가 어렵다 이번에는 임의로 설정) 직접 내가 할때는 평균디그라데이션 값을 고장기준이라 설정한다
failure_threshold

# Visualization of RUL Prediction
cl = bootlimit(tr_degScore, alpha=0.05, bootstrap=100) # 여기서 알파=(정규분포) 신뢰도수준의 alpha 이 외의 알파값은
                                                        # 가중치의 알파값(가까운거리면 높은 알파, 먼 거리면 낮은 알파)
    

plt.figure(figsize=(12,4))
plt.plot(ts_degScore, color='blue')

#(y=ax+b 에서의 a,b는처리됨) 디그라데이션 값=고장기준을 뚫는 시점을 예측한 고장 시점
#y= failure threshold 
#x=time
#구하는 것은 고장시점과 만난지점에서 x값을 구하는 것

#x= y-b/a = 예측한 고장시점
plt.xlim([0, 1200])
plt.ylim([min(ts_degScore), max(ts_degScore)*1.005])

x = np.arange(1400)
linear_curve = coef*x + intercept
plt.plot(linear_curve[0], color='darkcyan')

plt.axhline(y=failure_threshold, color='red')
plt.axhline(y=cl, color='red')
plt.show()

#맨 아래 빨간줄=cl(control limit), 위에빨간줄=고장기준(threshold), 파랑=디그라데이션 팩터, 초록=선형회귀로 구한 회귀선

# Result of RUL Prediction
predicted_failureTime= int((failure_threshold-intercept) / coef) # coef가 a라고 보면 됨필기한거있지
#(x=예측된 고장시점) 에 대해 전개를 한 것 x=y-b/a(failureTime=threshol-intercept/coef)
RUL = predicted_failureTime-t
#예측된 고장시점-현재시점(900으로 설정)


print('예측 잔여시점: %2.2f시점' % RUL)
print('예측 잔여수명: %2.4f일' % (RUL*10/60/24))
#베어링 데이터는 하나의 파일이 10분주기로 수집되어있어서 실제 시간으로 환산한 것
print()
print('실제 잔여시점: %2.2f시점' % (984-t))
print('실제 잔여수명: %2.4f일' % ((984-t)*10/60/24))