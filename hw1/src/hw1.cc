#include <cstdio>
#include <mpi.h>
#include <iostream>
#include <iomanip>  
#include <boost/sort/spreadsort/spreadsort.hpp>

void mergeData(float* localData, int localSize, float* neighborData, int neighborSize, float* tempBuffer, bool mergeFromMin) {
    if (mergeFromMin) {
        int localData_index = 0, neighborData_index = 0;

        for(int tempBuffer_index = 0; tempBuffer_index < localSize ; tempBuffer_index++){
            if(localData_index < localSize && (neighborData_index >= neighborSize || localData[localData_index] < neighborData[neighborData_index])){
                tempBuffer[tempBuffer_index] = localData[localData_index++];
            }else{
                tempBuffer[tempBuffer_index] = neighborData[neighborData_index++];
            }
        }
    } else {
        int localData_index = localSize - 1;
        int neighborData_index = neighborSize - 1;
        
        for(int tempBuffer_index = localSize - 1; tempBuffer_index >= 0 ; tempBuffer_index--){
            if(localData_index >= 0 && (neighborData_index < 0 || localData[localData_index] > neighborData[neighborData_index])){
                tempBuffer[tempBuffer_index] = localData[localData_index--];
            }else{
                tempBuffer[tempBuffer_index] = neighborData[neighborData_index--];
            }
        }
    }
}

int main(int argc, char **argv)
{
    if(argc != 4){
        std::cerr << "Usage: " << argv[0] << " <n> <input_file> <output_file>" << std::endl;
        return -1;
    }
    long long n = std::atoll(argv[1]); 
    char *input_filename = argv[2];
    char *output_filename = argv[3];

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 計時變數
    // double io_time_start, io_time_end, comm_time_start, comm_time_end, compute_time_start, compute_time_end;

    // I/O 讀取開始計時
    // io_time_start = MPI_Wtime();

    int each_rank_data = n / size;
    int remain = n % size;
    MPI_Offset offset = (each_rank_data * rank + std::min(remain, rank)) * sizeof(float);
    int each_rank_data_length = each_rank_data + (rank < remain);

    float *data = new float[each_rank_data_length];
    float* tempBuffer = new float[each_rank_data_length];

    MPI_File input_file;
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, offset, data, each_rank_data_length, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    // I/O 讀取結束計時
    // io_time_end = MPI_Wtime();

    // 計算開始計時
    // compute_time_start = MPI_Wtime();

    boost::sort::spreadsort::spreadsort(data, data + each_rank_data_length);

    int rank_neighbor[2], rank_neighbor_length[2];
    if(rank & 1){
        rank_neighbor[0] = rank - 1;
        rank_neighbor_length[0] = (rank_neighbor[0] < 0) ? 0 : each_rank_data + (rank_neighbor[0] < remain);
        rank_neighbor[1] = (rank + 1 == size) ? MPI_PROC_NULL : rank + 1;
        rank_neighbor_length[1] = (rank_neighbor[1] == MPI_PROC_NULL) ? 0 : each_rank_data + (rank_neighbor[1] < remain);
    }else{
        rank_neighbor[0] = (rank + 1 == size) ? MPI_PROC_NULL : rank + 1;
        rank_neighbor_length[0] = (rank_neighbor[0] == MPI_PROC_NULL) ? 0 : each_rank_data + (rank_neighbor[0] < remain);
        rank_neighbor[1] = (rank == 0) ? MPI_PROC_NULL : rank - 1;
        rank_neighbor_length[1] = (rank_neighbor[1] < 0) ? 0 : each_rank_data + (rank_neighbor[1] < remain);
    }

    int max_neighbor_length = std::max(rank_neighbor_length[0], rank_neighbor_length[1]);
    float* recvData = new float[max_neighbor_length];

    // 通訊開始計時
    // comm_time_start = MPI_Wtime();

    for (int i = 0; i < size + 1; i++) {
        bool isOddPhase = i % 2;

        int neighbor;
        int neighbor_length;

        if (isOddPhase) {
            neighbor = rank_neighbor[1];
            neighbor_length = rank_neighbor_length[1];
        } else {
            neighbor = rank_neighbor[0];
            neighbor_length = rank_neighbor_length[0];
        }

        if (neighbor == MPI_PROC_NULL) {
            continue;
        }

        MPI_Request send_request, recv_request;

        MPI_Isend(data, each_rank_data_length, MPI_FLOAT, neighbor, 0, MPI_COMM_WORLD, &send_request);
        MPI_Irecv(recvData, neighbor_length, MPI_FLOAT, neighbor, 0, MPI_COMM_WORLD, &recv_request);

        MPI_Wait(&send_request, MPI_STATUS_IGNORE);
        MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

        if (neighbor_length > 0) {
            bool mergeFromMin = (rank < neighbor);
            mergeData(data, each_rank_data_length, recvData, neighbor_length, tempBuffer, mergeFromMin);
            std::swap(data, tempBuffer);
        }
    }

    // 通訊結束計時
    // comm_time_end = MPI_Wtime();

    // 計算結束計時
    // compute_time_end = MPI_Wtime();

    // I/O 寫入開始計時
    // io_time_start = MPI_Wtime();

    MPI_File output_file;
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, offset, data, each_rank_data_length, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    // I/O 寫入結束計時
    // io_time_end += MPI_Wtime();

    delete[] data;
    delete[] tempBuffer;
    delete[] recvData;

    MPI_Finalize();

    // 設置小數點顯示格式
    // if (rank == 0) {
    //     std::cout << std::fixed << std::setprecision(2);
    //     std::cout << "I/O Time: " << (io_time_end - io_time_start) << " seconds" << std::endl;
    //     std::cout << "Communication Time: " << (comm_time_end - comm_time_start) << " seconds" << std::endl;
    //     std::cout << "Computation Time: " << (compute_time_end - compute_time_start) << " seconds" << std::endl;
    // }

    return 0;
}
