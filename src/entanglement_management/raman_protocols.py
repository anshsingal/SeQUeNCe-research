from src.protocol import Protocol
from src.message import Message
from src.kernel.event import Event
from src.kernel.process import Process
from matplotlib import pyplot as plt
from enum import Enum, auto
import numpy as np
import numpy.ma as ma
# import cupy as cp
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
    """ Sender's protocol"""
    def __init__(self, own, num_iterations, narrow_band_filter_bandwidth):
        self.own = own
        self.num_iterations = num_iterations
        # self.clock_power = clock_power
        self.narrow_band_filter_bandwidth = narrow_band_filter_bandwidth


    def start(self):
        """ Starts the protocol by scheduling the emit calls. """
        self.own.qchannels[self.own.signal_receiver].start_classical_communication()
        self.own.cchannels[self.own.signal_receiver].start_classical_communication()

        for i in range(self.num_iterations):
            last_emit_time = self.emit_event()

        print("last emit time:", last_emit_time)
        distance = self.own.qchannels["signal_receiver_1"].distance
        new_msg = ExpMsg(ExpMsgType.STOP, self.own.signal_receiver, distance = distance, last_emit_time = last_emit_time)

        process = Process(self.own, "send_message", [self.own.signal_receiver, new_msg])
        event = Event(self.own.timeline.now(), process)
        self.own.timeline.schedule(event)

    def emit_event(self):
        return self.own.parametric_source.schedule_emit()

    def received_message(self, src: str, message: Message):
        pass


class RamanTestReceiver(Protocol):
    """ Protocol at the receiver's end """
    def __init__(self, signal_node, idler_node, other_node: str, pulse_separation):
        self.signal_node = signal_node
        self.idler_node = idler_node
        self.other_node = other_node
        self.coincidence_times = []
        self.last_detection = 0

        self.half_hist_bin_width = int(pulse_separation/20)
        self.num_hist_bins_half = 45

        self.signal_buffer = []
        self.idler_buffer = []

    def start(self):
        pass
    
    def received_message(self, src: str, message: Message):
        """ Message signalling the receiver to stop the process, effectively ending the experiment. Post processing is initialized in this step."""
        assert message.msg_type == ExpMsgType.STOP

        distance = message.distance
        last_emit_time = message.last_emit_time

        stop_time = last_emit_time + 5*distance/(3*10**5) * 1e12

        process = Process(self, "stop_sim_and_process", [self.half_hist_bin_width * self.num_hist_bins_half])
        event = Event(stop_time, process)
        self.signal_node.timeline.schedule(event)        
        

    def stop_sim_and_process(self, hist_width):
        """This is the post processing step"""
        self.signal_node.timeline.stop()

        print("stopped sim and starting processing")
        

        # Call the detector files to start post-processing them. 
        self.signal_buffer = self.signal_node.detector.log_file
        self.idler_buffer = self.idler_node.detector.log_file
        
        self.signal_dead_time = self.signal_node.detector.dead_time
        self.idler_dead_time = self.idler_node.detector.dead_time
        
        prev_signal_detections = np.array([])
        prev_idler_detections = np.array([])

        prev_signal_dead_time = 0
        prev_idler_dead_time = 0

        # This segment removes the dead time detections (not handled during actual detection) and sorts the detections based on arrival time. 
        for i in sorted(map(int, self.signal_buffer)):
            print("sorting window", i)    
            signal_buffer, prev_signal_dead_time = self.sort_remove_dead_counts(self.signal_buffer[str(i)][:], self.signal_dead_time, prev_signal_dead_time)
            idler_buffer, prev_idler_dead_time = self.sort_remove_dead_counts(self.idler_buffer[str(i)][:], self.idler_dead_time, prev_idler_dead_time)
            print("sorted window", i)

            # The j_signal and j_idler are meant to account for the correlations between the detections in two different batches. 
            j_signal = 1
            j_idler = 1
            limit = signal_buffer[-1] - hist_width
            while signal_buffer[-j_signal] > limit:
                j_signal += 1
            while idler_buffer[-j_idler] > limit:
                j_idler += 1

            # Find the correlations between the idler and signal detections 
            self.get_correlations(np.append(prev_signal_detections, signal_buffer), np.append(prev_idler_detections, idler_buffer), hist_width, -j_signal, -j_idler)

            print("found correlations for window:", i)

            # Storing he last j_signal and j_idler elements to find correlations with the next batch
            prev_signal_detections = signal_buffer[-j_signal:]
            prev_idler_detections = idler_buffer[-j_idler:]


        self.get_correlations(prev_signal_detections, prev_idler_detections, hist_width, None, None)


        # Plottig and writing to CAR file.
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
        """ This method sorts the detections and removes the detections which are too close for the detector dead time"""
        
        # def GPU_sort(pulse_train):
        #     GPU_pulse_train = cp.asarray(pulse_train)
        #     GPU_sorted_pulse_train = cp.sort(GPU_pulse_train)
        #     return cp.asnumpy(GPU_sorted_pulse_train)

        # Sorting the array using a GPU for performance
        # sorted_pulse_train = GPU_sort(pulse_train)

        # Remove the detections which lie in the dead time of the previous batch
        for i in range(len(pulse_train)):
            if pulse_train[i] > prev_dead_time:
                break
        pulse_train = pulse_train[i:]

        # Removal of dark counts is done by JIT compiling the actual method and executing the kernel.
        @jit(parallel = True, nopython = True)
        def remove_dark_counts(pulse_train):
            mask = np.ones(len(pulse_train))
            i = 0
            while i<=len(pulse_train)-1:
                mask[i] = 0
                j = 1
                while len(pulse_train) > i+j and pulse_train[i+j] <= pulse_train[i] + dead_time:
                    j = j+1
                i = i + j
            return mask

        mask = remove_dark_counts(pulse_train)
        
        pulse_train = ma.masked_array(pulse_train, mask = mask)
        out = pulse_train[~pulse_train.mask]
        print("done with dark count removal")
        return out, out[-1] + dead_time


    def get_correlations(self, signal_buffer, idler_buffer, hist_width, j_signal, j_idler):
        """ Computes the correlations between the idler and signal detections. Assumed that the data is pre-processed."""

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