#include <cuda_runtime.h>
#include <stdint.h>

constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t BLOCK_X = 32;
constexpr uint32_t BLOCK_Y = 16;
constexpr uint32_t THREAD_X = 4;
constexpr uint32_t THREAD_Y = 4;

// Check keys to see if it is at the start/end of one tile's range in
// the full sorted list. If yes, write start/end of this tile.
// Run once per instanced (duplicated) Gaussian ID.
//识别排序后的键列表中每个tile范围的起始和结束位置
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, int2* ranges)  //L：排序后的键列表的长度  point_list_keys：key列表 输出ranges：存储每个tile范围开始和结束的位置 //64
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];  //读取当前key，从总提取tile id 64
	uint32_t currtile = key >> 32; //32
	if (idx == 0)
		ranges[currtile].x = 0;  //如果当前索引是列表的第一个元素，将当前tile范围的起始位置设置为 0
	else   //否则，对于其他索引，检查当前tile是否与前一个tile相同
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32; //32
		if (currtile != prevtile)    //如果当前tile与前一个tile不同，则更新前一个tile范围的结束位置和当前tile范围的起始位置
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

__forceinline__ __device__ void pixel_shader(float3& C, float& T, float2 pixf, float2 xy, float4 con_o, float3 rgb)
{
	float2 d = { xy.x - pixf.x, xy.y - pixf.y };
	//float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
	float power = con_o.w + con_o.x * d.x * d.x + con_o.z * d.y * d.y + con_o.y * d.x * d.y;
	float alpha;
	asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(alpha) : "f"(power));
	"\"";
	//alpha = min(0.99f, alpha);
	C.x += rgb.x * (alpha * T);
	C.y += rgb.y * (alpha * T);
	C.z += rgb.z * (alpha * T);
	T -= alpha * T;
}

__forceinline__ __device__ uint8_t encode(float x)
{
	return (uint8_t)min(max(0.0f, x * 255.0f), 255.0f);
}

__forceinline__ __device__ uint8_t write_color(uchar3* __restrict__ out_color,
	float3 bg_color, uint2 pix, int width, int height, float3 C, float T)
{
	if (pix.x < width && pix.y < height)
	{
		int pix_id = width * pix.y + pix.x;
		if (T < 0.0001f)
		{
			T = 0.0f;
		}
		out_color[pix_id].x = encode(C.x + T * bg_color.x);
		out_color[pix_id].y = encode(C.y + T * bg_color.y);
		out_color[pix_id].z = encode(C.z + T * bg_color.z);
	}
}

__global__ void renderCUDA(
	const int2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int width, int height, int horizontal_blocks,
	const float2* __restrict__ points_xy_image,
	const float3* __restrict__ features,
	const float* __restrict__ depths,
	const float4* __restrict__ conic_opacity,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	int2 range = ranges[blockIdx.y * horizontal_blocks + blockIdx.x];
	uint32_t lane_id = threadIdx.y * blockDim.x + threadIdx.x;
	const void* data = nullptr;
	int scale = 0;
	switch (lane_id)
	{
	case 0:
		data = point_list;
		scale = 1;
		break;
	case 8:
		data = &points_xy_image->x;
		scale = 2;
		break;
	case 9:
		data = &points_xy_image->y;
		scale = 2;
		break;
	case 16:
		data = &features->x;
		scale = 3;
		break;
	case 17:
		data = &features->y;
		scale = 3;
		break;
	case 18:
		data = &features->z;
		scale = 3;
		break;
	case 24:
		data = &conic_opacity->x;
		scale = 4;
		break;
	case 25:
		data = &conic_opacity->y;
		scale = 4;
		break;
	case 26:
		data = &conic_opacity->z;
		scale = 4;
		break;
	case 27:
		data = &conic_opacity->w;
		scale = 4;
		break;
	}
	scale *= 4; // sizeof(int), sizeof(float)
	if (range.x >= range.y)
	{
		return;
	}
	int point_id = range.x;
	int coll_id = point_list[point_id];
	float2 xy = points_xy_image[coll_id];
	float3 rgb = features[coll_id];
	float4 con_o = conic_opacity[coll_id];
	coll_id = point_list[point_id + 1];
	if (lane_id == 0)
	{
		coll_id = point_id + 2;
	}

	// uint2 pix = { blockIdx.x * BLOCK_X + threadIdx.x, blockIdx.y * BLOCK_Y + threadIdx.y };
	uint2 pix[THREAD_Y][THREAD_X];
#pragma unroll
	for (uint32_t i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (uint32_t j = 0; j < THREAD_X; j++)
		{
			pix[i][j] = {
				blockIdx.x * BLOCK_X + threadIdx.x * THREAD_X + j,
				blockIdx.y * BLOCK_Y + threadIdx.y * THREAD_Y + i
			};
		}
	}

	// float2 pixf = { (float)pix.x, (float)pix.y };
	float2 pixf[THREAD_Y][THREAD_X];
#pragma unroll
	for (uint32_t i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (uint32_t j = 0; j < THREAD_X; j++)
		{
			pixf[i][j] = { (float)pix[i][j].x, (float)pix[i][j].y };
		}
	}

	float T[THREAD_Y][THREAD_X];
#pragma unroll
	for (uint32_t i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (uint32_t j = 0; j < THREAD_X; j++)
		{
			T[i][j] = 1.0f;
		}
	}
	float3 C[THREAD_Y][THREAD_X] = { 0.0f };
	bool done = true;
	float buf;
	coll_id *= scale;
	do {
		if (data != nullptr)
		{
			buf = *reinterpret_cast<const float*>(reinterpret_cast<const char*>(data) + coll_id);
		}

#pragma unroll
		for (uint32_t i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (uint32_t j = 0; j < THREAD_X; j++)
			{
				pixel_shader(C[i][j], T[i][j], pixf[i][j], xy, con_o, rgb);
			}
		}

		done = true;
#pragma unroll
		for (uint32_t i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (uint32_t j = 0; j < THREAD_X; j++)
			{
				done = done && T[i][j] < 0.0001f;
			}
		}
		coll_id = __shfl_sync(~0, __float_as_uint(buf), 0);
		if (lane_id == 0)
		{
			coll_id = point_id + 3;
		}
		xy = {
			__shfl_sync(~0, buf, 8),
			__shfl_sync(~0, buf, 9)
		};
		rgb = {
			__shfl_sync(~0, buf, 16),
			__shfl_sync(~0, buf, 17),
			__shfl_sync(~0, buf, 18)
		};
		con_o = {
			__shfl_sync(~0, buf, 24),
			__shfl_sync(~0, buf, 25),
			__shfl_sync(~0, buf, 26),
			__shfl_sync(~0, buf, 27)
		};
		coll_id *= scale;
	} while (__any_sync(~0, ++point_id < range.y && !done));
#pragma unroll
	for (uint32_t i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (uint32_t j = 0; j < THREAD_X; j++)
		{
			write_color(out_color, bg_color, pix[i][j], width, height, C[i][j], T[i][j]);
		}
	}
}


void render(int num_rendered,
	int width, int height,
	float* points_xy, float* depths, float* rgb, float* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int* ranges,
	float3 bg_color, char* out_color)
{
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	cudaMemsetAsync(ranges, 0, sizeof(int2) * grid.x * grid.y);

	identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(
		num_rendered,
		gaussian_keys_sorted,
		(int2*)ranges);

	int horizontal_blocks = (width + BLOCK_X - 1) / BLOCK_X;
	renderCUDA<<<grid, dim3(8, 4, 1)>>>(
		(int2*)ranges,
		gaussian_values_sorted,
		width, height,
		horizontal_blocks,
		(float2*)points_xy,
		(float3*)rgb,
		depths,
		(float4*)conic_opacity,
		bg_color,
		(uchar3*)out_color);
}
