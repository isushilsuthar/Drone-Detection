

import sys
import threading
import queue
import audioop
import math
import pyaudio
import numpy as np
from gcc_phat import gcc_phat


RATE = 48000
FRAMES = int(RATE / 4)

window = np.hanning(FRAMES)

sound_speed = 343.2
distance = 0.14

max_tau = distance / sound_speed
direction_n = int(max_tau * RATE)


class DOA:
    def __init__(self):
        self.pyaudio_instance = pyaudio.PyAudio()
        self.queue = queue.Queue()
        self.event = threading.Event()

    def start(self, quit_event=None, show=None):
        stream = self.pyaudio_instance.open(
            rate=RATE,
            frames_per_buffer=FRAMES,
            format=pyaudio.paInt16,
            channels=2,
            input=True,
            # output_device_index=1,
            stream_callback=self._callback)

        self.event.clear()
        if not quit_event:
            quit_event = threading.Event()

        phat = [0] * (2 * direction_n + 1)
        while not (quit_event.is_set() or self.event.is_set()):
            try:
                data = self.queue.get()

                buf = np.fromstring(data, dtype='int16')
                tau, cc = gcc_phat(buf[0::2] * window, buf[1::2] * window, fs=RATE, max_tau=max_tau, interp=1)
                theta = math.asin(tau / max_tau) * 180 / math.pi
                print('\ntheta: {}'.format(int(theta)))

                for i, v in enumerate(cc):
                    phat[i] = int(v * 512)

                if show:
                    show(phat)
                # print [l for l in level]
            except KeyboardInterrupt:
                break

        stream.close()

    def _callback(self, in_data, frame_count, time_info, status):
        self.queue.put(in_data)

        return None, pyaudio.paContinue


def main():
    from PySide import QtGui
    from bar_widget import BarWidget

    app = QtGui.QApplication(sys.argv)
    widget = BarWidget()
    widget.setWindowTitle('Direction Of Arrival')
    widget.show()

    doa = DOA()

    quit_event = threading.Event()
    thread = threading.Thread(target=doa.start, args=(quit_event, widget.setBars))
    thread.start()

    app.exec_()

    quit_event.set()
    thread.join()

if __name__ == '__main__':
    main()

