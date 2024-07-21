#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

void preprocess(int P,
	float* orig_points, float* shs, float* opacities, float* cov3Ds,
	int width, int height, glm::vec3 position, glm::mat3 rotation, float focal_x, float focal_y, float zFar, float zNear,
	float* points_xy, float* depths, float* rgb, float* conic_opacity,
	uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted,
	int* curr_offset);

void sort_gaussian(int num_rendered,
	int width, int height, int block_x, int block_y,
	char* list_sorting_space, size_t sorting_size,
	uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted);

void render(int num_rendered, 
	int width, int height,
	float* points_xy, float* depths, float* rgb, float* conic_opacity,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int* ranges,
	float3 bg_color, char* out_color);
