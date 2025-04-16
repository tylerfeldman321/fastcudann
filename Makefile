all: 5_nn_single_layer_graph_timing_less_sync.exe

profile: 

5_nn_single_layer_graph_timing_less_sync.exe: 5_nn_single_layer_graph_timing_less_sync.cu
	nvcc 5_nn_single_layer_graph_timing_less_sync.cu -L /usr/local/cuda/lib -lcudart -o 5_nn_single_layer_graph_timing_less_sync.exe -run

clean:
	rm 5_nn_single_layer_graph_timing_less_sync.exe
