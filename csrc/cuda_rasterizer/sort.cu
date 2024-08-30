#include "../ops.h"

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

namespace flashgs {
namespace {

static uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

} // namespace

void sort_gaussian(int num_rendered,
    int width, int height, int block_x, int block_y,
	char* list_sorting_space, size_t sorting_size,
	uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted, cudaStream_t stream)
{
	dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y, 1);
	auto status = cub::DeviceRadixSort::SortPairs(
		list_sorting_space, sorting_size,
		gaussian_keys_unsorted, gaussian_keys_sorted,
		gaussian_values_unsorted, gaussian_values_sorted,
		num_rendered, 0, 32 + getHigherMsb(grid.x * grid.y), stream);
    if (status != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

size_t get_sort_buffer_size(int num_rendered, cudaStream_t stream)
{
    size_t sort_buffer_size = 0;
	cub::DeviceRadixSort::SortPairs<uint64_t, uint32_t>(
		nullptr, sort_buffer_size,
		nullptr, nullptr,
		nullptr, nullptr, num_rendered, 0, sizeof(uint64_t) * 8, stream);
    return sort_buffer_size;
}

} // namespace flashgs