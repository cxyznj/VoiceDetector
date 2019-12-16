import pyaudio
import wave
import numpy as np
import os
import math
import scipy.io.wavfile as wav
import Gvoicesource
import shutil
from multiprocessing import Process, Queue
import time
import interface

RESPEAKER_RATE = 44100
RESPEAKER_CHANNELS = 8 
RESPEAKER_WIDTH = 2
# run getDeviceInfo.py to get index
RESPEAKER_INDEX = 2  # refer to input device id
CHUNK = 1024
RECORD_SECONDS = 1
qcount = 0

#
def save_record(WAVE_FILENAME, frames, p):
    wf = wave.open(WAVE_FILENAME, 'wb')
    # set how many channels you extract
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
    wf.setframerate(RESPEAKER_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    rate, mic = wav.read(WAVE_FILENAME)


def recording(p, WAVE_OUTPUT_FOLDNAME):
    stream = p.open(
                rate=RESPEAKER_RATE,
                format=p.get_format_from_width(RESPEAKER_WIDTH),
                channels=RESPEAKER_CHANNELS,
                input=True,
                input_device_index=RESPEAKER_INDEX,)

    frames1 = []
    frames2 = []
    frames3 = []
    frames4 = []

    for i in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        # extract channel 0 data from 8 channels, if you want to extract channel 1, please change to [1::8]
        # 4 channel for microphone, 2 channels for the playback, 2 channels are not used
        mic1 = np.fromstring(data, dtype=np.int16)[0::8]
        frames1.append(mic1.tostring())
        mic2 = np.fromstring(data, dtype=np.int16)[1::8]
        frames2.append(mic2.tostring())
        mic3 = np.fromstring(data, dtype=np.int16)[2::8]
        frames3.append(mic3.tostring())
        mic4 = np.fromstring(data, dtype=np.int16)[3::8]
        frames4.append(mic4.tostring())

    stream.stop_stream()
    stream.close()

    os.mkdir(WAVE_OUTPUT_FOLDNAME)

    save_record(WAVE_OUTPUT_FOLDNAME+"//mic1.wav", frames1, p)
    save_record(WAVE_OUTPUT_FOLDNAME+"//mic2.wav", frames2, p)
    save_record(WAVE_OUTPUT_FOLDNAME+"//mic3.wav", frames3, p)
    save_record(WAVE_OUTPUT_FOLDNAME+"//mic4.wav", frames4, p)


def recording_fetch(p, max_count, q):
    global qcount
    while qcount < max_count:
        recording(p, 'record' + str(qcount))
        qcount += 1
        q.put(qcount-1)


def detecting(WAVE_OUTPUT_FOLDNAME):
    angle, distance = Gvoicesource.get_voice_source(WAVE_OUTPUT_FOLDNAME+"//mic1.wav", WAVE_OUTPUT_FOLDNAME+"//mic2.wav", WAVE_OUTPUT_FOLDNAME+"//mic3.wav", WAVE_OUTPUT_FOLDNAME+"//mic4.wav")
   
    if angle != None:
        angle = round(angle, 3)
        distance = round(distance, 3)
        print("象限" + str(quad), str(angle) + '°', str(distance)+'cm')
        return angle, distance
    else:
        print("Fail to detect...")
        return None, None


if __name__ == "__main__":
    p = pyaudio.PyAudio()
    recordq = Queue()
    max_count = 50
    recorder = Process(target=recording_fetch, args=(p, max_count, recordq))
    recorder.start()
    
    # 加入图形界面
    interfaceq = Queue()
    inter = Process(target=interface.interface_fetch, args=(0, interfaceq))
    inter.start()

    bgtime = time.time()
    
    while True:
        if not recordq.empty():
            i = recordq.get()
            if i >= max_count:
                recorder.join()
                break
            else:
                angle, distance = detecting('record' + str(i))
                # WARNING:这一行是否需要删掉待测试
                # shutil.rmtree('record' + str(i))
                durtime = time.time() - bgtime
                quad = 1
                if angle > 90.0:
                    quad = 2
                    angle = 180 - angle
                xpoint = int(math.cos(angle/180*3.14159) * distance * 0.5)
                ypoint = int(math.sin(angle/180*3.14159) * distance * 0.5)
                interfaceq.put([durtime, xpoint, ypoint])
        else:
            time.sleep(0.1)

    p.terminate()
    exit()