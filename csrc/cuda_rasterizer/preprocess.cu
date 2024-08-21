#ifndef CUDA_VERSION
#define CUDA_VERSION 8000
#endif

#define GLM_FORCE_CUDA
#include "../glm/glm.hpp"

constexpr uint32_t BLOCK_X = 32;
constexpr uint32_t BLOCK_Y = 16;
constexpr uint32_t WARP_SIZE = 32;
constexpr float log2e = 1.4426950216293334961f;
constexpr float ln2 = 0.69314718055f;

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	glm::mat4 viewmatrix,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	p_view = transformPoint4x3(p_orig, (float*)&viewmatrix);
	return p_view.z > 0.2f;
}

__forceinline__ __device__ void getRect(const float2 p, int width, int height, int2& rect_min, int2& rect_max, dim3 grid)
{
	rect_min = {
		min((int)grid.x, max((int)0, (int)((p.x - width) / (float)BLOCK_X))),
		min((int)grid.y, max((int)0, (int)((p.y - height) / (float)BLOCK_Y)))
	};
	rect_max = {
		min((int)grid.x, max((int)0, (int)((p.x + width) / (float)BLOCK_X) + 1)),
		min((int)grid.y, max((int)0, (int)((p.y + height) / (float)BLOCK_Y) + 1))
	};
}

// Forward version of 2D covariance matrix computation
__forceinline__ __device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy,
	const float* cov3D, glm::mat4 viewmatrix, bool is_fisheye)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, (float*)&viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J;
	if (is_fisheye)
	{
		//float eps = 0.001f;
		float eps = 0.0000001f;
		float x2 = t.x * t.x + eps;
		float y2 = t.y * t.y;
		float xy = t.x * t.y;
		float x2y2 = x2 + y2;
		float len_xy = length(glm::vec2({ t.x, t.y })) + eps;
		float x2y2z2_inv = 1.f / (x2y2 + t.z * t.z);

		float b = glm::atan(len_xy, t.z) / len_xy / x2y2;
		float a = t.z * x2y2z2_inv / (x2y2);
		J = glm::mat3(
			focal_x * (x2 * a + y2 * b), focal_x * xy * (a - b), -focal_x * t.x * x2y2z2_inv,
			focal_y * xy * (a - b), focal_y * (y2 * a + x2 * b), -focal_y * t.y * x2y2z2_inv,
			0, 0, 0
		);
	}
	else
	{
		J = glm::mat3(
			focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
			0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
			0, 0, 0);
	}

	glm::mat3 W = glm::mat3(
		((float*)&viewmatrix)[0], ((float*)&viewmatrix)[4], ((float*)&viewmatrix)[8],
		((float*)&viewmatrix)[1], ((float*)&viewmatrix)[5], ((float*)&viewmatrix)[9],
		((float*)&viewmatrix)[2], ((float*)&viewmatrix)[6], ((float*)&viewmatrix)[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__forceinline__ __device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const glm::vec3* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	auto sh = shs + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;
	return glm::max(result, 0.0f);
}

__forceinline__ __device__ bool segment_intersect_ellipse(float a, float b, float c, float d, float l, float r)
{
	float delta = b * b - 4.0f * a * c;
	if (delta >= 0.0f)
	{
		//float sqrt_delta = __fsqrt_ru(delta);
		float t1 = (l - d) * (2.0f * a) + b;
		float t2 = (r - d) * (2.0f * a) + b;
		/*if (t1 <= sqrt_delta && t2 >= -sqrt_delta)
		{
			return true;
		}*/
		if ((t1 <= 0.0f || t1 * t1 <= delta) && (t2 >= 0.0f || t2 * t2 <= delta))
		{
			return true;
		}
	}
	return false;
}

__forceinline__ __device__ bool block_intersect_ellipse(int2 pix_min, int2 pix_max, float2 center, float3 conic, float power)
{
	float a, b, c, delta, dx, dy;
	int lambda1, lambda2;
	float w = 2.0f * power;

	dx = center.x - pix_min.x;
	a = conic.z;
	b = -2.0f * conic.y * dx;
	c = conic.x * dx * dx - w;
	/*delta = b * b - 4.0f * a * c;
	if (delta >= 0.0f)
	{
		float sqrt_delta = __fsqrt_ru(delta);
		lambda1 = (-b + sqrt_delta) / (2.0f * a) + center.y;
		lambda2 = (-b - sqrt_delta) / (2.0f * a) + center.y;
		if (pix_min.y <= lambda1 && pix_max.y >= lambda2)
		{
			return true;
		}
	}*/
	if (segment_intersect_ellipse(a, b, c, center.y, pix_min.y, pix_max.y))
	{
		return true;
	}

	dx = center.x - pix_max.x;
	a = conic.z;
	b = -2.0f * conic.y * dx;
	c = conic.x * dx * dx - w;
	/*delta = b * b - 4.0f * a * c;
	if (delta >= 0.0f)
	{
		float sqrt_delta = __fsqrt_ru(delta);
		lambda1 = (-b + sqrt_delta) / (2.0f * a) + center.y;
		lambda2 = (-b - sqrt_delta) / (2.0f * a) + center.y;
		if (pix_min.y <= lambda1 && pix_max.y >= lambda2)
		{
			return true;
		}
	}*/
	if (segment_intersect_ellipse(a, b, c, center.y, pix_min.y, pix_max.y))
	{
		return true;
	}

	dy = center.y - pix_min.y;
	a = conic.x;
	b = -2.0f * conic.y * dy;
	c = conic.z * dy * dy - w;
	/*delta = b * b - 4.0f * a * c;
	if (delta >= 0.0f)
	{
		float sqrt_delta = __fsqrt_ru(delta);
		lambda1 = (-b + sqrt_delta) / (2.0f * a) + center.x;
		lambda2 = (-b - sqrt_delta) / (2.0f * a) + center.x;
		if (pix_min.x <= lambda1 && pix_max.x >= lambda2)
		{
			return true;
		}
	}*/
	if (segment_intersect_ellipse(a, b, c, center.x, pix_min.x, pix_max.x))
	{
		return true;
	}

	dy = center.y - pix_max.y;
	a = conic.x;
	b = -2.0f * conic.y * dy;
	c = conic.z * dy * dy - w;
	/*delta = b * b - 4.0f * a * c;
	if (delta >= 0.0f)
	{
		float sqrt_delta = __fsqrt_ru(delta);
		lambda1 = (-b + sqrt_delta) / (2.0f * a) + center.x;
		lambda2 = (-b - sqrt_delta) / (2.0f * a) + center.x;
		if (pix_min.x <= lambda1 && pix_max.x >= lambda2)
		{
			return true;
		}
	}*/
	if (segment_intersect_ellipse(a, b, c, center.x, pix_min.x, pix_max.x))
	{
		return true;
	}

	return false;
}

__forceinline__ __device__ bool block_contains_center(int2 pix_min, int2 pix_max, float2 center)
{
	return center.x >= pix_min.x && center.x <= pix_max.x && center.y >= pix_min.y && center.y <= pix_max.y;
}

template<bool is_fisheye = false>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const float* opacities,
	const glm::vec3* shs,
	glm::mat4 viewmatrix,
	glm::mat4 projmatrix,
	glm::vec3 cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	float2* points_xy,
	float* depths,
	float* cov3Ds,
	glm::vec3* rgb,
	float4* conic_opacity,
	int* curr_offset,
	uint64_t* gaussian_keys_unsorted,   //用于存储键值对的数组 /64
	uint32_t* gaussian_values_unsorted,  //用于存储键值对的数组
	const dim3 grid)
{
	int lane = threadIdx.y * blockDim.x + threadIdx.x;
	int warp_id = blockIdx.x * blockDim.z + threadIdx.z;
	int idx_vec = warp_id * WARP_SIZE + lane;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	bool point_valid = false;
	int width = 0;
	int height = 0;
	float3 p_view;
	float2 point_xy;
	float3 conic;
	float opacity;
	float power;
	float log2_opacity;
	int2 rect_min;
	int2 rect_max;
	if (idx_vec < P)
	{
		do {
			// Perform near culling, quit if outside.
			float3 p_orig = { orig_points[3 * idx_vec], orig_points[3 * idx_vec + 1], orig_points[3 * idx_vec + 2] };
			p_view = transformPoint4x3(p_orig, (float*)&viewmatrix);
			if (p_view.z <= 0.2f)
				break;
			opacity = opacities[idx_vec];
			if (255.0f * opacity < 1.0f)
				break;

			//3d高斯球的中心点投影
			// Transform point by projecting
			float3 p_proj;
			if (is_fisheye)
			{
				float xy_len = glm::length(glm::vec2(p_view.x, p_view.y)) + 0.0001f;
				float theta = glm::atan(xy_len, p_view.z);
				if (glm::abs(theta) > 3.14f * 0.403f)
					break;
				p_proj.x = 2.0f * p_view.x * focal_x * theta / (xy_len * W);
				p_proj.y = 2.0f * p_view.y * focal_y * theta / (xy_len * H);
				p_proj.z = 0.0f;
			}
			else
			{
				float4 p_hom = transformPoint4x4(p_orig, (float*)&projmatrix);
				float p_w = 1.0f / (p_hom.w + 0.0000001f);
				p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
			}

			//3d高斯投影在平面上的椭圆
			// Compute 2D screen-space covariance matrix
			float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3Ds + idx_vec * 6, viewmatrix, is_fisheye);

			// Invert covariance (EWA algorithm)
			float det = (cov.x * cov.z - cov.y * cov.y);
			float det_inv = 1.f / det;
			conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

			log2_opacity = __log2f(opacity);
			power = ln2 * 8.0f + ln2 * log2_opacity;
			width = (int)(1.414214f * __fsqrt_ru(cov.x * power) + 1.0f);
			height = (int)(1.414214f * __fsqrt_ru(cov.z * power) + 1.0f);

			//包围盒覆盖了哪些tile，要用的值存储在中间结果中
			point_xy = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
			getRect(point_xy, width, height, rect_min, rect_max, grid);
			point_valid = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) > 0;
		} while (false);
	}

	bool single_tile = point_valid && (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 1;
	int any_valid = __ballot_sync(~0, single_tile);
	if (single_tile)
	{
		int2 pix_min = { rect_min.x * BLOCK_X, rect_min.y * BLOCK_Y };
		int2 pix_max = { pix_min.x + BLOCK_X - 1, pix_min.y + BLOCK_Y - 1 };
		bool valid = block_contains_center(pix_min, pix_max, point_xy) ||
			block_intersect_ellipse(pix_min, pix_max, point_xy, conic, power);
		if (valid)
		{
			uint64_t key = rect_min.y * grid.x + rect_min.x;   //得到tile对应本张图片的tile_id 64
			key <<= 32;
			key |= __float_as_uint(p_view.z);  //该高斯球对应的深度 depth_id  32
			int offset = atomicAdd(curr_offset, 1);
			gaussian_keys_unsorted[offset] = key;    //key
			gaussian_values_unsorted[offset] = idx_vec;  //高斯id
		}
		point_valid = false;
	}

	// Generate no key/value pair for invisible Gaussians
	int multi_tiles = __ballot_sync(~0, point_valid);
	for (int i = 0; i < WARP_SIZE; i++)
	{
		if (multi_tiles & (1 << i))
		{
			// Find this Gaussian's offset in buffer for writing keys/values.
			int idx = warp_id * 32 + i;
			float2 my_point_xy = {
				__shfl_sync(~0, point_xy.x, i),
				__shfl_sync(~0, point_xy.y, i)
			};
			float3 my_conic = {
				__shfl_sync(~0, conic.x, i),
				__shfl_sync(~0, conic.y, i),
				__shfl_sync(~0, conic.z, i),
			};
			float2 my_rect_min = {
				__shfl_sync(~0, rect_min.x, i),
				__shfl_sync(~0, rect_min.y, i)
			};
			float2 my_rect_max = {
				__shfl_sync(~0, rect_max.x, i),
				__shfl_sync(~0, rect_max.y, i)
			};
			float my_depth = __shfl_sync(~0, p_view.z, i);
			float my_power = __shfl_sync(~0, power, i);

			// For each tile that the bounding rect overlaps, emit a 
			// key/value pair. The key is |  tile ID  |      depth      |,
			// and the value is the ID of the Gaussian. Sorting the values 
			// with this key yields Gaussian IDs in a list, such that they
			// are first sorted by tile and then by depth. 
			__syncwarp();
			bool vertex_valid = false;
			for (int y0 = my_rect_min.y; y0 < my_rect_max.y; y0 += blockDim.y)   //循环迭代tile范围，为每个tile生成键值对
			{
				int y = y0 + threadIdx.y;
				for (int x0 = my_rect_min.x; x0 < my_rect_max.x; x0 += blockDim.x)
				{
					int x = x0 + threadIdx.x;
					bool valid = y < my_rect_max.y && x < my_rect_max.x;

					if (valid)
					{
						int2 pix_min = { x * BLOCK_X, y * BLOCK_Y };
						int2 pix_max = { pix_min.x + BLOCK_X - 1, pix_min.y + BLOCK_Y - 1 };
						valid = block_contains_center(pix_min, pix_max, my_point_xy) ||
							block_intersect_ellipse(pix_min, pix_max, my_point_xy, my_conic, my_power);
					}

					int mask = __ballot_sync(~0, valid);
					if (mask == 0)
					{
						continue;
					}
					int my_offset;
					if (lane == 0)
					{
						my_offset = atomicAdd(curr_offset, __popc(mask));
					}
					vertex_valid = true;
					int count = __popc(mask & ((1 << lane) - 1));
					uint64_t key = y * grid.x + x;   //得到tile对应本张图片的tile_id 64
					key <<= 32;
					key |= __float_as_uint(my_depth);  //该高斯球对应的深度 depth_id  32
					my_offset = __shfl_sync(~0, my_offset, 0);
					if (valid)
					{
						gaussian_keys_unsorted[my_offset + count] = key;    //key
						gaussian_values_unsorted[my_offset + count] = idx;  //高斯id
					}
				}
			}
			if (__any_sync(~0, vertex_valid))
			{
				any_valid |= 1 << i;
			}
		}
	}
	if (any_valid)
	{
		if (any_valid & (1 << lane))
		{
			glm::vec3 color = computeColorFromSH(idx_vec, D, M, (glm::vec3*)orig_points, cam_pos, shs, nullptr);
			rgb[idx_vec] = color;
		}
		depths[idx_vec] = p_view.z;
		points_xy[idx_vec] = point_xy;
		conic_opacity[idx_vec] = { (-0.5f * log2e) * conic.x, -log2e * conic.y, (-0.5f * log2e) * conic.z, log2_opacity };
	}
}

glm::mat4 getViewMatrix(glm::vec3 position, glm::mat3 rotation)
{
	return glm::mat4(
		glm::vec4(rotation[0], 0.0f),
		glm::vec4(rotation[1], 0.0f),
		glm::vec4(rotation[2], 0.0f),
		glm::vec4(rotation * -position, 1.0f));
}

glm::mat4 getProjectionMatrix(int width, int height, glm::vec3 position, glm::mat3 rotation, float focal_x, float focal_y, float zFar, float zNear)
{
	float top = height / (2.0f * focal_y) * zNear;
	float bottom = -top;
	float right = width / (2.0f * focal_x) * zNear;
	float left = -right;

	glm::mat4 P;
	memset(&P, 0, sizeof P);
	float z_sign = 1.0f;

	P[0][0] = 2.0f * zNear / (right - left);
	P[1][1] = 2.0f * zNear / (top - bottom);
	P[0][2] = (right + left) / (right - left);
	P[1][2] = (top + bottom) / (top - bottom);
	P[3][2] = z_sign;
	P[2][2] = z_sign * zFar / (zFar - zNear);
	P[2][3] = -(zFar * zNear) / (zFar - zNear);
	return glm::transpose(P) * getViewMatrix(position, rotation);
}

void preprocess(int P,
	float* orig_points, float* shs, float* opacities, float* cov3Ds,
	int width, int height, glm::vec3 position, glm::mat3 rotation, float focal_x, float focal_y, float zFar, float zNear,
	float* points_xy, float* depths, float* rgb, float* conic_opacity,
	uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted,
	int* curr_offset)
{
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	glm::mat4 viewmatrix = getViewMatrix(position, rotation);
	glm::mat4 projmatrix = getProjectionMatrix(width, height, position, rotation, focal_x, focal_y, zFar, zNear);
	float tan_fovx = width / (2.0f * focal_x);
	float tan_fovy = height / (2.0f * focal_y);

	int D = 3;
	int M = 16;

	preprocessCUDA<<<(P + 127) / 128, dim3(8, 4, 4)>>>(
		P, D, M,
		orig_points,
		opacities,
		(glm::vec3*)shs,
		viewmatrix,
		projmatrix,
		position,
		width, height,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		(float2*)points_xy,
		depths,
		cov3Ds,
		(glm::vec3*)rgb,
		(float4*)conic_opacity,
		curr_offset,
		gaussian_keys_unsorted,
		gaussian_values_unsorted,
		grid);
}
