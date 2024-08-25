#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include "glm/glm.hpp"

constexpr int WARP_SIZE = 32;

#define CHECK_CUDA(x)                                                                   \
	{                                                                                   \
		cudaError_t status = x;                                                         \
		if (status != cudaSuccess) {                                                    \
			fprintf(stderr, "%s\nline = %d\n", cudaGetErrorString(status), __LINE__);   \
			exit(1);                                                                    \
		}                                                                               \
	}

union cov3d_t
{
    float2 f2[3];
    float s[6];
};

union shs_deg3_t
{
    float4 f4[12];
    glm::vec3 v3[16];
};

void preprocess(int P,
	glm::vec3* positions, shs_deg3_t* shs, float* opacities, cov3d_t* cov3Ds,
	int width, int height, int block_x, int block_y,
	glm::vec3 cam_position, glm::mat3 cam_rotation,
	float focal_x, float focal_y, float zFar, float zNear,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted,
	int* curr_offset);

void sort_gaussian(int num_rendered,
	int width, int height, int block_x, int block_y,
	char* list_sorting_space, size_t sorting_size,
	uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted);

size_t get_sort_buffer_size(int num_rendered);

void render_16x16(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int2* ranges, float3 bg_color, uchar3* out_color);

void render_32x16(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int2* ranges, float3 bg_color, uchar3* out_color);

void render_32x32(int num_rendered,
	int width, int height,
	float2* points_xy, float4* rgb_depth, float4* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int2* ranges, float3 bg_color, uchar3* out_color);
