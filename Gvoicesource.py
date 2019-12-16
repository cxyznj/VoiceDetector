import scipy
import numpy as np
import scipy.io.wavfile as wav
import math
import random

# model: -----mic1-----mic2-----mic3-----mic4-----


# Parameter: sig: input signal A; refsig: input signal B
# Return: signal offset, CC function
def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=1):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]
 
    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
 
    if all(np.abs(R)) == True:
        cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
    else:
        cc = np.fft.irfft(R, n =(interp * n))
 
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)
 
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
 
    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift
 
    tau = shift / float(interp * fs)
    
    return tau, cc 

    
# Parameter: slice1: one slice in signal A; slice2: one slice in signal B; sample_rate: signal sample rate; max_offset: minimum offset in this circumstance
# Return: 直线的偏向, 距离差
def caculate_distancedif(slice1, slice2, sample_rate, max_offset):
    min_dif, cc = gcc_phat(slice1, slice2)
    if max_offset <= abs(min_dif) < 1.1 * max_offset:
        if min_dif < 0:
            min_dif = -max_offset
        else:
            min_dif = max_offset
    elif abs(min_dif) >= 1.1 * max_offset:
        return None, None
    # 直线的斜率，当互相关算出来相位差为正时表示slice1比slice2声音时间早，直线应向slice1偏
    symbol = 1
    if min_dif < 0:
        symbol = -1
    # distance以厘米为单位
    distancedif = abs(min_dif * 1.0 / sample_rate) * 34000
    return symbol, distancedif


# Parameter: slice1/2: 第一、二个信号, sample_rate: 信号采样率; c = 这对麦克风的焦距
# Return: 基于这对麦克风计算出的双曲线渐近线(单支)的斜率k
def caculate_k(slice1, slice2, sample_rate, c = 5.0):
    # 根据几何关系计算出
    max_offset = 18.8 /5 * c
    symbol, distancedif = caculate_distancedif(slice1, slice2, sample_rate, max_offset)
    if symbol == None:
        return None
        
    a = distancedif / 2.0
    if abs(a) < 1e-7:
        if a >= 0:
            a = 1e-7
        else:
            a = -1e-7

    b = math.sqrt(c * c - a * a)
    
    k = b / a * symbol
    
    return k


# Parameter: slice1/2/3/4: 四个麦克风收到的信号; sample_rate: 信号采样率
# Return: 预测的声源角度和距离
def cal_voice_source(slice1, slice2, slice3, slice4, sample_rate):
    # group1: mic1/mic3; group2: mic2/mic4;
    # group3: mic1/mic2; group4: mic2/mic3; group5: mic3/mic4;
    # gourp6: mic1/mic4
    # kx + y = b

    k1 = caculate_k(slice1, slice3, sample_rate, 5.0)
    k2 = caculate_k(slice2, slice4, sample_rate, 5.0)
    k3 = caculate_k(slice1, slice2, sample_rate, 2.5)
    k4 = caculate_k(slice2, slice3, sample_rate, 2.5)
    k5 = caculate_k(slice3, slice4, sample_rate, 2.5)
    k6 = caculate_k(slice1, slice4, sample_rate, 10.0)

    if k1 == None or k2 == None or k3 == None or k4 == None or k5 == None or k6 == None:
        return None, None, None

    k1 = -k1
    k2 = -k2
    k3 = -k3
    k4 = -k4
    k5 = -k5
    k6 = -k6

    A = np.array([[k1, k2, k3, k4, k5, k6], [1, 1, 1, 1, 1, 1]])
    b = np.array([[(-2.5)*k1, 2.5*k2, (-5)*k3, 0*k4, 5*k5, 0*k6]])

    ATA = np.dot(A, A.T)
    ATB = np.dot(b, A.T)

    result = np.dot(ATB, np.linalg.inv(ATA))

    x = result[0][0]
    y = result[0][1]

    angle = math.atan(abs(y)/abs(x)) / (2 * 3.14159) * 360

    if x < 0:
        angle = 180 - angle

    distance = math.sqrt(y * y + x * x)
    #if x < 1e-7 and y < 1e-7:
    #    return None, None, None

    return angle, distance
    

# 外部调用函数
# Parameter: 四个麦克风收到的声音的文件路径
# Return: 预测出的声源的角度和距离
def get_voice_source(Filename1, Filename2, Filename3, Filename4):
    rate1, mic1 = wav.read(Filename1)
    rate2, mic2 = wav.read(Filename2)
    rate3, mic3 = wav.read(Filename3)
    rate4, mic4 = wav.read(Filename4)

    assert(rate1 == rate2 == rate3 == rate4)
    sample_rate = rate1

    voicelen = min(len(mic1), len(mic2), len(mic3), len(mic4))

    angle, distance = cal_voice_source(mic1, mic2, mic3, mic4, sample_rate)
    
    return angle, distance
    
    
if __name__ == "__main__":
    foldname = input("Please input fold name:")

    rate1, mic1 = wav.read(foldname + "/" + "mic1.wav")
    rate2, mic2 = wav.read(foldname + "/" + "mic2.wav")
    rate3, mic3 = wav.read(foldname + "/" + "mic3.wav")
    rate4, mic4 = wav.read(foldname + "/" + "mic4.wav")

    assert(rate1 == rate2 == rate3 == rate4)
    sample_rate = rate1

    voicelen = min(len(mic1), len(mic2), len(mic3), len(mic4))

    angle, distance = cal_voice_source(mic1, mic2, mic3, mic4, sample_rate)
    
    print(angle, distance)
