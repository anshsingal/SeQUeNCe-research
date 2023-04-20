# from src.topology.node import Node
from src.protocol import Protocol
from src.message import Message
from src.kernel.event import Event
from src.kernel.process import Process
from matplotlib import pyplot as plt
from enum import Enum, auto
import numpy as np
import numpy.ma as ma
import cupy as cp
from numba import jit

class ExpMsgType(Enum):    
    STOP = auto()

class ExpMsg(Message):

    def __init__(self, msg_type: ExpMsgType, receiver: str, **kwargs):
        super().__init__(msg_type, receiver)

        if msg_type is ExpMsgType.STOP:
            self.distance = kwargs.get("distance")
            self.last_emit_time = kwargs.get("last_emit_time")

class RamanTestSender(Protocol):
    def __init__(self, own, num_iterations, clock_power, narrow_band_filter_bandwidth):
        self.own = own
        self.num_iterations = num_iterations
        self.clock_power = clock_power
        self.narrow_band_filter_bandwidth = narrow_band_filter_bandwidth
        # own.protocols.append(self)
        # self.other_nodes = other_nodes

    def start(self):
        self.own.qchannels[self.own.signal_receiver].start_clock(self.clock_power, self.narrow_band_filter_bandwidth)

        for i in range(self.num_iterations):
            last_emit_time = self.emit_event()

        print("last emit time:", last_emit_time)
        distance = self.own.qchannels["signal_receiver"].distance
        new_msg = ExpMsg(ExpMsgType.STOP, self.own.signal_receiver, distance = distance, last_emit_time = last_emit_time)

        process = Process(self.own, "send_message", [self.own.signal_receiver, new_msg])
        event = Event(self.own.timeline.now(), process)
        self.own.timeline.schedule(event)

    def emit_event(self):
        return self.own.parametric_source.schedule_emit()

    def received_message(self, src: str, message: Message):
        pass


class RamanTestReceiver(Protocol):
    def __init__(self, signal_node, idler_node, other_node: str, pulse_separation):
        # signal_node.protocols.append(self)
        # idler_node.protocols.append(self)
        self.signal_node = signal_node
        self.idler_node = idler_node
        self.other_node = other_node
        # self.first_detection = True
        # self.first_detection_time = 0
        self.coincidence_times = []
        # self.stop_receiving = False
        self.last_detection = 0

        self.half_hist_bin_width = int(pulse_separation/20)
        self.num_hist_bins_half = 45

        self.signal_buffer = []
        self.idler_buffer = []

    def start(self):
        # start clock and emissions
        pass
    
    def received_message(self, src: str, message: Message):
        assert message.msg_type == ExpMsgType.STOP


        distance = message.distance
        last_emit_time = message.last_emit_time

        stop_time = last_emit_time + 5*distance/(3*10**5) * 1e12

        # print("stop time:", stop_time, "distance:", distance)

        process = Process(self, "stop_sim_and_process", [self.half_hist_bin_width * self.num_hist_bins_half])
        event = Event(stop_time, process)
        self.signal_node.timeline.schedule(event)


        

        # self.stop_receiving = True
        # self.get_correlations(self.half_hist_bin_width * self.num_hist_bins_half)
        
        # n, bins, patches = plt.hist(self.coincidence_times, range(-self.half_hist_bin_width * self.num_hist_bins_half, self.half_hist_bin_width * self.num_hist_bins_half, self.half_hist_bin_width))# , range(-28125, 28126, 625)
        # plt.yscale('log')
        # plt.show()
        
        

    def stop_sim_and_process(self, hist_width):
        self.signal_node.timeline.stop()

        # return # Not doing data processing

        print("stopped sim and starting processing")
        
        self.signal_buffer = self.signal_node.detector.log_file
        self.idler_buffer = self.idler_node.detector.log_file
        
        self.signal_dead_time = self.signal_node.detector.dead_time
        self.idler_dead_time = self.idler_node.detector.dead_time
        # print("detector processing step")
        
        prev_signal_detections = np.array([])
        prev_idler_detections = np.array([])

        prev_signal_dead_time = 0
        prev_idler_dead_time = 0

        for i in sorted(map(int, self.signal_buffer)):
            print("sorting window", i)    
            # print("length of signal buffer:", len(self.signal_buffer[str(i)][:]), "idler_buffer:", len(self.idler_buffer[str(i)][:]))
            signal_buffer, prev_signal_dead_time = self.sort_remove_dead_counts(self.signal_buffer[str(i)][:], self.signal_dead_time, prev_signal_dead_time)
            idler_buffer, prev_idler_dead_time = self.sort_remove_dead_counts(self.idler_buffer[str(i)][:], self.idler_dead_time, prev_idler_dead_time)
            # print("length of dead_time removed signal buffer:", len(self.signal_buffer[str(i)][:]), "idler_buffer:", len(self.idler_buffer[str(i)][:]))
            print("sorted window", i)

            # print("sorted signal_buffer:", signal_buffer)
            # print("sorted idler buffer:", idler_buffer)

            # print("sorted and deadcount removed signal buffer:", signal_buffer)
            # print("sorted and deadcount removed idler buffer:", idler_buffer)

            j_signal = 1
            j_idler = 1
            limit = signal_buffer[-1] - hist_width
            while signal_buffer[-j_signal] > limit:
                j_signal += 1
            while idler_buffer[-j_idler] > limit:
                j_idler += 1

            self.get_correlations(np.append(prev_signal_detections, signal_buffer), np.append(prev_idler_detections, idler_buffer), hist_width, -j_signal, -j_idler)

            print("found correlations for window:", i)

            prev_signal_detections = signal_buffer[-j_signal:]
            prev_idler_detections = idler_buffer[-j_idler:]



        # Uncomment this to turn on correlations again.
        self.get_correlations(prev_signal_detections, prev_idler_detections, hist_width, None, None)

        # print("hist_width:", hist_width, "correlations:", self.coincidence_times)

        plt.hist(self.coincidence_times, range(-hist_width,hist_width, self.half_hist_bin_width))# , range(-28125, 28126, 625)
        n, edges = np.histogram(self.coincidence_times, bins = int(2*hist_width/self.half_hist_bin_width), range = (-hist_width, hist_width))
        n = list(n)
        matched_correlation = max(n)
        n.remove(matched_correlation)
        unmatched_correlations = []
        for i in range(4):
            unmatched_correlations.append(max(n))
            n.remove(unmatched_correlations[-1])
        CAR = matched_correlation/(sum(unmatched_correlations)/len(unmatched_correlations))
        print("CAR:", CAR)
        file1 = open("CAR_Data.txt", "a")  # append mode
        file1.write(f"{CAR}\n")
        file1.close()
        plt.yscale('log')
        plt.show()

        self.signal_buffer.close()
        self.idler_buffer.close()

            

    def sort_remove_dead_counts(self, pulse_train, dead_time, prev_dead_time):
        
        def GPU_sort(pulse_train):
            GPU_pulse_train = cp.asarray(pulse_train)
            GPU_sorted_pulse_train = cp.sort(GPU_pulse_train)
            return cp.asnumpy(GPU_sorted_pulse_train)

        sorted_pulse_train = GPU_sort(pulse_train)
        # print("done sorting")
        # original_size = len(pulse_train)

        for i in range(len(pulse_train)):
            if pulse_train[i] > prev_dead_time:
                break
        sorted_pulse_train = sorted_pulse_train[i:]

        # print("starting dark count removal")
        
        # kept_detections = np.array([])
        @jit(parallel = True, nopython = True)
        def remove_dark_counts(sorted_pulse_train):
            mask = np.ones(len(sorted_pulse_train))
            i = 0
            while i<=len(sorted_pulse_train)-1:
                mask[i] = 0
                # print("dark count removal:", i, len(sorted_pulse_train))
                j = 1
                while len(sorted_pulse_train) > i+j and sorted_pulse_train[i+j] <= sorted_pulse_train[i] + dead_time:
                    j = j+1
                i = i + j
            return mask

        mask = remove_dark_counts(sorted_pulse_train)
        
        # np.delete(sorted_pulse_train, indices_to_remove)
        sorted_pulse_train = ma.masked_array(sorted_pulse_train, mask = mask)
        out = sorted_pulse_train[~sorted_pulse_train.mask]
        print("done with dark count removal")
        return out, out[-1] + dead_time



    def get_correlations(self, signal_buffer, idler_buffer, hist_width, j_signal, j_idler):
        last_detection_time = 0
        # You iterate over the signal buffer. You look for the idler detections ahead of the present 
        # signal detection which fall inside the hist_width
        for i in signal_buffer[:j_signal]:
            j = last_detection_time
            
            # The last idler detectiuon may be smaller than the present signal detection. So, we need to bring the idler detection to atleast the 
            # present signal detection and we can start counting correlations from there.  
            while j < len(idler_buffer) and idler_buffer[j] < i:
                j += 1

            # We save this for the next signal detection since while iterating over idler detections for this signal detection, we may go over the 
            # ones which could also form correlations with the next signal detection.  
            last_detection_time = j

            # Once your present idler detection >= present signal detection, you can start going ahead in idler detections until the hist_width (ahead
            # of the present signal detection) has been reached. 
            while j < len(idler_buffer) and idler_buffer[j] < i + hist_width:
                self.coincidence_times.append(idler_buffer[j] - i)
                j += 1

        # Very similar for the idler case as in the signal case before this. 
        last_detection_time = 0
        for i in idler_buffer[:j_idler]:
            j = last_detection_time
            while j < len(signal_buffer) and signal_buffer[j] < i:
                j += 1
            last_detection_time = j
            while j < len(signal_buffer) and signal_buffer[j] < i + hist_width:
                if not i-signal_buffer[j] == 0:
                    self.coincidence_times.append(i-signal_buffer[j])
                j += 1

    # def get_correlations(self, signal_buffer, idler_buffer, hist_width, j_signal, j_idler):
    #     last_detection_time = 0
    #     for i in signal_buffer:
    #         j = last_detection_time
    #         while j < len(idler_buffer) and idler_buffer[j] < i:
    #             j += 1
    #         last_detection_time = j
    #         while j < len(idler_buffer) and idler_buffer[j] < i + hist_width:
    #             self.coincidence_times.append(idler_buffer[j] - i)
    #             j += 1

    #     last_detection_time = 0
    #     for i in idler_buffer:
    #         j = last_detection_time
    #         while j < len(signal_buffer) and signal_buffer[j] < i:
    #             j += 1
    #         last_detection_time = j
    #         while j < len(signal_buffer) and signal_buffer[j] < i + hist_width:
    #             self.coincidence_times.append(i-signal_buffer[j])
    #             j += 1    


        