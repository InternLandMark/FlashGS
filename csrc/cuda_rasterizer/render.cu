#include "../ops.h"

namespace flashgs {
namespace {

// Check keys to see if it is at the start/end of one tile's range in
// the full sorted list. If yes, write start/end of this tile.
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, int2* ranges)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32; //32
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32; //32
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

__forceinline__ __device__ float fast_ex2_ftz_f32(float x)
{
	float y;
	asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
	return y;
}

__forceinline__ __device__ void pixel_shader(float3& C, float& T, float2 pixf, float2 xy, float4 con_o, float3 rgb)
{
	float2 d = { xy.x - pixf.x, xy.y - pixf.y };
	//float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
	float power = con_o.w + con_o.x * d.x * d.x + con_o.z * d.y * d.y + con_o.y * d.x * d.y;
	float alpha;
	asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(alpha) : "f"(power));
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
	float3 bg_color, int2 pix, int width, int height, float3 C, float T)
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

struct render_load_info
{
	const void* data[FLASHGS_WARP_SIZE] = { nullptr };
	int lg2_scale[FLASHGS_WARP_SIZE] = { 0 };

	render_load_info(const uint32_t* point_list, const float2* points_xy, const float4* rgb_depth, const float4* conic_opacity)
	{
		for (int lane = 0; lane < 32; lane++)
		{
			switch (lane)
			{
			case 0:
				data[lane] = point_list;
				lg2_scale[lane] = 2;
				break;
			case 4:
				data[lane] = point_list;
				lg2_scale[lane] = 2;
				break;
			case 8:
				data[lane] = &points_xy->x;
				lg2_scale[lane] = 3;
				break;
			case 9:
				data[lane] = &points_xy->y;
				lg2_scale[lane] = 3;
				break;
				break;
			case 12:
				data[lane] = &points_xy->x;
				lg2_scale[lane] = 3;
				break;
			case 13:
				data[lane] = &points_xy->y;
				lg2_scale[lane] = 3;
				break;
			case 16:
				data[lane] = &rgb_depth->x;
				lg2_scale[lane] = 4;
				break;
			case 17:
				data[lane] = &rgb_depth->y;
				lg2_scale[lane] = 4;
				break;
			case 18:
				data[lane] = &rgb_depth->z;
				lg2_scale[lane] = 4;
				break;
			case 19:
				data[lane] = &rgb_depth->w;
				lg2_scale[lane] = 4;
				break;
			case 20:
				data[lane] = &rgb_depth->x;
				lg2_scale[lane] = 4;
				break;
			case 21:
				data[lane] = &rgb_depth->y;
				lg2_scale[lane] = 4;
				break;
			case 22:
				data[lane] = &rgb_depth->z;
				lg2_scale[lane] = 4;
				break;
			case 23:
				data[lane] = &rgb_depth->w;
				lg2_scale[lane] = 4;
				break;
			case 24:
				data[lane] = &conic_opacity->x;
				lg2_scale[lane] = 4;
				break;
			case 25:
				data[lane] = &conic_opacity->y;
				lg2_scale[lane] = 4;
				break;
			case 26:
				data[lane] = &conic_opacity->z;
				lg2_scale[lane] = 4;
				break;
			case 27:
				data[lane] = &conic_opacity->w;
				lg2_scale[lane] = 4;
				break;
			case 28:
				data[lane] = &conic_opacity->x;
				lg2_scale[lane] = 4;
				break;
			case 29:
				data[lane] = &conic_opacity->y;
				lg2_scale[lane] = 4;
				break;
			case 30:
				data[lane] = &conic_opacity->z;
				lg2_scale[lane] = 4;
				break;
			case 31:
				data[lane] = &conic_opacity->w;
				lg2_scale[lane] = 4;
				break;
			}
		}
	}
};

__forceinline__ __device__ void get_gaussian_features(float2& xy, float3& rgb, float4& con_o, float buf, int offset)
{
	xy = {
		__shfl_sync(~0, buf, 8 + offset),
		__shfl_sync(~0, buf, 9 + offset)
	};
	rgb = {
		__shfl_sync(~0, buf, 16 + offset),
		__shfl_sync(~0, buf, 17 + offset),
		__shfl_sync(~0, buf, 18 + offset)
	};
	con_o = {
		__shfl_sync(~0, buf, 24 + offset),
		__shfl_sync(~0, buf, 25 + offset),
		__shfl_sync(~0, buf, 26 + offset),
		__shfl_sync(~0, buf, 27 + offset)
	};
}

template<int BLOCK_X, int BLOCK_Y, int THREAD_X, int THREAD_Y>
__global__ void renderCUDA(
	const int2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int width, int height, int x_blocks,
	const float2* __restrict__ points_xy,
	const float4* __restrict__ rgb_depth,
	const float4* __restrict__ conic_opacity,
	render_load_info info,
	float3 bg_color,
	uchar3* __restrict__ out_color)
{
	int2 range = ranges[blockIdx.y * x_blocks + blockIdx.x];
	int lane = threadIdx.y * blockDim.x + threadIdx.x;
	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];

	// uint2 pix = { blockIdx.x * BLOCK_X + threadIdx.x, blockIdx.y * BLOCK_Y + threadIdx.y };
	int2 pix[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			pix[i][j] = {
				(int)blockIdx.x * BLOCK_X + (int)threadIdx.x * THREAD_X + j,
				(int)blockIdx.y * BLOCK_Y + (int)threadIdx.y * THREAD_Y + i
			};
		}
	}

	// float2 pixf = { (float)pix.x, (float)pix.y };
	float2 pixf[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			pixf[i][j] = { (float)pix[i][j].x, (float)pix[i][j].y };
		}
	}

	float T[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			T[i][j] = 1.0f;
		}
	}

	float3 C[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			C[i][j] = { 0.0f, 0.0f, 0.0f };
		}
	}

	int point_id = range.x;
	if (point_id < range.y)
	{
		int offset;
		float2 xy;
		float3 rgb;
		float4 con_o;
		if (lane == 0)
		{
			offset = point_id + 2;
		}
		else if (lane == 4)
		{
			offset = point_id + 3;
		}
		else if ((lane & 4) == 0 && point_id + 1 < range.y)
		{
			offset = point_list[point_id + 0];
		}
		else if (point_id + 2 < range.y)
		{
			offset = point_list[point_id + 1];
		}
		const float* ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(data) + ((int64_t)offset << lg2_scale));
		float buf;
		bool load_enable = data != nullptr;
		if (lane == 0)
		{
			load_enable = load_enable && point_id + 2 < range.y;
		}
		else if (lane == 4)
		{
			load_enable = load_enable && point_id + 3 < range.y;
		}
		else if ((lane & 4) == 0)
		{
			load_enable = load_enable && point_id + 0 < range.y;
		}
		else
		{
			load_enable = load_enable && point_id + 1 < range.y;
		}
		if (load_enable)
		{
			buf = __ldg(ptr); // 0: point_list[point_id + 2], 4: point_list[point_id + 3], 8: features[point_list[point_id + 0]], 12: features[point_list[point_id + 1]]
		}

		load_enable = data != nullptr;

		bool done = false;
		while (__any_sync(~0, point_id + 5 < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0)
			{
				offset = point_id + 4;
			}
			if (lane == 4)
			{
				offset = point_id + 5;
			}

#ifdef _DEBUG
			if (lane == 0)
			{
				printf("point_id = %d\n", point_id);
			}
#endif
			float ldg_buf;
			ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(data) + ((int64_t)offset << lg2_scale));
			if (load_enable)
			{
				ldg_buf = __ldg(ptr); // 0: point_list[point_id + 4], 4: point_list[point_id + 5], 8: features[point_list[point_id + 2]], 12: features[point_list[point_id + 3]]
#ifdef _DEBUG
				if (lane == 0 && __float_as_int(ldg_buf) != point_list[point_id + 4])
				{
					printf("error1\n");
				}
				else if (lane == 4 && __float_as_int(ldg_buf) != point_list[point_id + 5])
				{
					printf("error2\n");
				}
				else if (lane == 8 && ldg_buf != points_xy[point_list[point_id + 2]].x)
				{
					printf("error3\n");
				}
				else if (lane == 12 && ldg_buf != points_xy[point_list[point_id + 3]].x)
				{
					printf("error4\n");
				}
#endif
			}

			get_gaussian_features(xy, rgb, con_o, buf, 0);
#ifdef _DEBUG
			if (lane == 3 && xy.x != points_xy[point_list[point_id + 0]].x)
			{
				printf("error5\n");
			}
#endif

	#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
	#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					pixel_shader(C[i][j], T[i][j], pixf[i][j], xy, con_o, rgb);
				}
			}

			get_gaussian_features(xy, rgb, con_o, buf, 4);
#ifdef _DEBUG
			if (lane == 3 && xy.x != points_xy[point_list[point_id + 1]].x)
			{
				printf("error6\n");
			}
#endif

	#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
	#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					pixel_shader(C[i][j], T[i][j], pixf[i][j], xy, con_o, rgb);
				}
			}

			done = true;
	#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
	#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					done = done && T[i][j] < 0.0001f;
				}
			}

			point_id += 2;
			buf = ldg_buf;
		}
		while (__any_sync(~0, point_id < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 4);
			if (lane == 0)
			{
				offset = point_id + 4;
			}
			if (lane == 4)
			{
				offset = point_id + 5;
			}

			if (lane == 0)
			{
				load_enable = load_enable && point_id + 4 < range.y;
			}
			else if (lane == 4)
			{
				load_enable = load_enable && point_id + 5 < range.y;
			}
			else if ((lane & 4) == 0)
			{
				load_enable = load_enable && point_id + 2 < range.y;
			}
			else
			{
				load_enable = load_enable && point_id + 3 < range.y;
			}

#ifdef _DEBUG
			if (lane == 0)
			{
				printf("point_id = %d\n", point_id);
			}
#endif
			float ldg_buf;
			ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(data) + ((int64_t)offset << lg2_scale));
			if (load_enable)
			{
				ldg_buf = __ldg(ptr); // 0: point_list[point_id + 4], 4: point_list[point_id + 5], 8: features[point_list[point_id + 2]], 12: features[point_list[point_id + 3]]
#ifdef _DEBUG
				if (lane == 0 && __float_as_int(ldg_buf) != point_list[point_id + 4])
				{
					printf("error1\n");
				}
				else if (lane == 4 && __float_as_int(ldg_buf) != point_list[point_id + 5])
				{
					printf("error2\n");
				}
				else if (lane == 8 && ldg_buf != points_xy[point_list[point_id + 2]].x)
				{
					printf("error3\n");
				}
				else if (lane == 12 && ldg_buf != points_xy[point_list[point_id + 3]].x)
				{
					printf("error4\n");
				}
#endif
			}

			get_gaussian_features(xy, rgb, con_o, buf, 0);
#ifdef _DEBUG
			if (lane == 3 && xy.x != points_xy[point_list[point_id + 0]].x)
			{
				printf("error5\n");
			}
#endif

#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					pixel_shader(C[i][j], T[i][j], pixf[i][j], xy, con_o, rgb);
				}
			}

			if (point_id + 1 >= range.y)
				break;

			get_gaussian_features(xy, rgb, con_o, buf, 4);
#ifdef _DEBUG
			if (lane == 3 && xy.x != points_xy[point_list[point_id + 1]].x)
			{
				printf("error6\n");
			}
#endif

#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					pixel_shader(C[i][j], T[i][j], pixf[i][j], xy, con_o, rgb);
				}
			}

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					done = done && T[i][j] < 0.0001f;
				}
			}
			point_id += 2;
			buf = ldg_buf;
		}
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				write_color(out_color, bg_color, pix[i][j], width, height, C[i][j], T[i][j]);
			}
		}
	}
	else
	{
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				int pix_x = blockIdx.x * BLOCK_X + threadIdx.x * THREAD_X + j;
				int pix_y = blockIdx.y * BLOCK_Y + threadIdx.y * THREAD_Y + i;
				int pix_id = width * pix_y + pix_x;
				out_color[pix_id].x = encode(bg_color.x);
				out_color[pix_id].y = encode(bg_color.y);
				out_color[pix_id].z = encode(bg_color.z);
			}
		}
	}
}

template<int BLOCK_X, int BLOCK_Y>
void render(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int2* ranges, float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	cudaMemsetAsync(ranges, 0, sizeof(int2) * grid.x * grid.y, stream);

    // Identify start and end of per-tile workloads in sorted list
    identifyTileRanges<<<(num_rendered + 255) / 256, 256, 0, stream>>>(
        num_rendered,
        gaussian_keys_sorted,
        ranges);

    // Let each tile blend its range of Gaussians independently in parallel
    renderCUDA<BLOCK_X, BLOCK_Y, BLOCK_X / 8, BLOCK_Y / 4><<<grid, dim3(8, 4, 1), 0, stream>>>(
        ranges,
        gaussian_values_sorted,
        width, height, grid.x,
        points_xy,
        rgb_depth,
        conic_opacity,
        render_load_info(gaussian_values_sorted, points_xy, rgb_depth, conic_opacity),
        bg_color,
        out_color);
}

} // namespace

void render_16x16(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int2* ranges, float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
    render<16, 16>(num_rendered, width, height, points_xy, rgb_depth, conic_opacity,
	    gaussian_keys_sorted, gaussian_values_sorted, ranges, bg_color, out_color, stream);
}

void render_32x16(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int2* ranges, float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
    render<32, 16>(num_rendered, width, height, points_xy, rgb_depth, conic_opacity,
	    gaussian_keys_sorted, gaussian_values_sorted, ranges, bg_color, out_color, stream);
}

void render_32x32(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int2* ranges, float3 bg_color, uchar3* out_color, cudaStream_t stream)
{
    render<32, 32>(num_rendered, width, height, points_xy, rgb_depth, conic_opacity,
	    gaussian_keys_sorted, gaussian_values_sorted, ranges, bg_color, out_color, stream);
}

} // namespace flashgs