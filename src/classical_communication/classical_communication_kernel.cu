// extern "C" __global__ 
// void pairwise_addition_kernel(bool* in_array, long* out_array, int limit){
//     int thread_ID = blockDim.x * blockIdx.x + threadIdx.x;
//     if (thread_ID < limit){
//         printf("limit is: %d\n", limit);
//         if (in_array[thread_ID*2] == 0){
//             if (in_array[thread_ID*2+1] == 0){
//                 printf("we have 00\n");
//             }
//             else {
//                 printf("we have 01\n");
//             }
//         }
//         else{
//             if (in_array[thread_ID*2+1] == 0){
//                 printf("we have 10\n");
//             }
//             else {
//                 printf("we have 11\n");
//             }
//         }


//         // printf("bit is: %d\n", in_array[thread_ID]);
//         // printf("size of int is: %ld, thread_ID is: %d with sum %ld\n", sizeof(), thread_ID, in_array[0]);
//         out_array[thread_ID] = in_array[thread_ID*2] + in_array[thread_ID*2+1];
//     }
// }
#include <curand_kernel.h>
#include <curand.h>
extern "C" __global__ 
void Raman_Kernel(bool* bits, long* noise_photons, int limit, double* classical_powers, 
double raman_coefficient, double narrow_band_filter_bandwidth, double quantum_channel_attenuation, double pulse_width,
double classical_channel_attenuation, double window_size, double h, double c, double quantum_channel_wavelength, double classical_rate, 
double distance, double collection_probability, double quantum_channel_index, double classical_channel_index, long max_raman_photons_per_pulse){
    
    
    int symbol_number = blockDim.x * blockIdx.x + threadIdx.x;
    
    // if (thread_ID == 0){
    //     printf("limit : %d\n", limit);
    //     printf("raman_coefficient : %.5e\n", raman_coefficient);
    //     printf("narrow_band_filter_bandwidth : %.5e\n", narrow_band_filter_bandwidth);
    //     printf("quantum_channel_attenuation : %.5e\n", quantum_channel_attenuation);
    //     printf("pulse_width : %.5e\n", pulse_width);
    //     printf("classical_channel_attenuation : %.5e\n", classical_channel_attenuation);
    //     printf("window_size : %.5e\n", window_size);
    //     printf("h : %.5e\n", h);
    //     printf("c : %.5e\n", c);
    //     printf("quantum_channel_wavelength : %.5e\n", quantum_channel_wavelength);
    //     printf("classical_rate : %.5e\n", classical_rate);
    //     printf("distance : %.5e\n", distance);
    //     printf("bits : %d\n", bits[1]);
    //     printf("noise_photons : %.5e\n", noise_photons[0]);
    //     printf("classical_powers : %.5e\n", classical_powers[0]);
    // }

    // printf("through here 1\n");

    if (symbol_number < limit){
        double location, probability_of_transmission;
        double decision;
        double detection_time;
        double classical_speed = c/classical_channel_index, quantum_speed = c/quantum_channel_index;
        int num_writes = 0;
        curandState_t state;
        curand_init(clock64(), symbol_number, 0, &state);


        double raman_energy = window_size * classical_powers[bits[symbol_number*2]*2 + bits[symbol_number*2+1]] * raman_coefficient * narrow_band_filter_bandwidth * (exp(-quantum_channel_attenuation * pulse_width) - exp(-classical_channel_attenuation * pulse_width)) / (classical_channel_attenuation - quantum_channel_attenuation);
        double mean_num_photons = (raman_energy / (h * c / quantum_channel_wavelength));
        int num_photons_added = curand_poisson(&state, mean_num_photons);

        for (int i = 0; i<num_photons_added; i++) {
            location = distance * curand_uniform(&state);
            probability_of_transmission = exp(-quantum_channel_attenuation*distance)*(exp((quantum_channel_attenuation - classical_channel_attenuation)*location)-1) / (exp(-classical_channel_attenuation*distance) - exp(-quantum_channel_attenuation*distance));
            decision = pow(0., floor(curand_uniform(&state)/(probability_of_transmission*collection_probability)));
            // How are we making decisions here: 0^0 is 1 in C. So, whatever is our probability,
            // we multiply its reciprocla with a uniform sample. So, you have U*(1/p). Call this z.
            // Now, one unit in z has a ratio of 1:p. Hence, if the region between 0->1 has a probability
            // of p. Now, the the GIF of this region = 0. So, 0^0 = 1 and anything else = 0. So, this 
            // becomes our decision. If 0^(GIF(U*(1/p))) == 1: Accept, else, its value = 0 and hence, reject. 
            // Here, we are considering both, the probability of transmission and the probability of the photon
            // actually getting detected.           
            detection_time = (symbol_number/(classical_rate/2) + (location*1000 / classical_speed + (distance-location)*1000 / quantum_speed) * 1e12);

            noise_photons[symbol_number * max_raman_photons_per_pulse + num_writes] = detection_time;
            num_writes += 1*decision;
        }
        noise_photons[symbol_number * max_raman_photons_per_pulse + num_writes] = 0.;
    }
}