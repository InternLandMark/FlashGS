#include "ops.h"

#include <torch/extension.h>

#include <fstream>
#include <iostream>
#include <string>

struct VertexStorage
{
    glm::vec3 position;
    glm::vec3 normal;
    float shs[48];
    float opacity;
    glm::vec3 scale;
    glm::vec4 rotation;
};

void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
    // Create scaling matrix
    glm::mat3 S = glm::mat3(1.0f);
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;

    // Normalize quaternion to get valid rotation
    glm::vec4 q = rot;// / glm::length(rot);
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    // Compute rotation matrix from quaternion
    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    glm::mat3 M = S * R;

    // Compute 3D world covariance matrix Sigma
    glm::mat3 Sigma = glm::transpose(M) * M;

    // Covariance is symmetric, only store upper right
    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> loadPly_torch(const std::string& name)
{
    std::ifstream file(name, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error(std::string("Failed to open file: ") + name);
    }
    bool end_header = false;
    int numVertex = 0;
    std::string s;
    while (!file.eof())
    {
        file >> s;
        if (s == "vertex")
        {
            file >> numVertex;
            if (numVertex <= 0)
            {
                throw std::runtime_error("Vertex number is not positive");
            }
        }
        else if (s == "end_header")
        {
            end_header = true;
            file.get();
            break;
        }
    }
    if (!end_header)
    {
        throw std::runtime_error("Cannot find end of header");
    }
    torch::Device device(torch::kCPU);
    torch::TensorOptions options(torch::kFloat32);
    torch::Tensor positionTensor = torch::empty({numVertex,3}, options.device());
    torch::Tensor shsTensor = torch::empty({numVertex,48}, options.device(device));
    torch::Tensor opacityTensor = torch::empty({numVertex}, options.device(device));
    torch::Tensor cov3dTensor = torch::empty({numVertex,6}, options.device(device));
    auto position = (glm::vec3*)positionTensor.contiguous().data_ptr<float>();
    auto shs = (glm::vec3*)shsTensor.contiguous().data_ptr<float>();
    auto opacity = opacityTensor.contiguous().data_ptr<float>();
    auto cov3d = cov3dTensor.contiguous().data_ptr<float>();
    for (int i = 0; i < numVertex; i++)
    {
        VertexStorage buf;
        file.read(reinterpret_cast<char*>(&buf), sizeof(VertexStorage));
        position[i] = buf.position;
        constexpr int SH_N = 16;
        //memcpy(&shs[i * SH_N], buf.shs, 48 * sizeof(float));
        shs[i * SH_N] = { buf.shs[0], buf.shs[1], buf.shs[2] };
        for (auto j = 1; j < SH_N; j++)
        {
            shs[i * SH_N + j] = { buf.shs[(j - 1) + 3], buf.shs[(j - 1) + SH_N + 2], buf.shs[(j - 1) + SH_N * 2 + 1] };
        }
        opacity[i] = 1.0f / (1.0f + std::exp(-buf.opacity));
        buf.scale.x = std::exp(buf.scale.x);
        buf.scale.y = std::exp(buf.scale.y);
        buf.scale.z = std::exp(buf.scale.z);
        buf.rotation = glm::normalize(buf.rotation);
        computeCov3D(buf.scale, 1.0, buf.rotation, &cov3d[i * 6]);
    }
    return std::make_tuple(numVertex, positionTensor, shsTensor, opacityTensor, cov3dTensor);
}

void preprocess_torch(
	torch::Tensor& orig_points, torch::Tensor& shs, torch::Tensor& opacities, torch::Tensor& cov3Ds,
	int width, int height, int block_x, int block_y,
    torch::Tensor& position, torch::Tensor& rotation,
    float focal_x, float focal_y, float zFar, float zNear,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_unsorted, torch::Tensor& gaussian_values_unsorted,
	torch::Tensor& curr_offset)
{
    auto position_data = position.contiguous().data_ptr<float>();
    auto rotation_data = rotation.contiguous().data_ptr<float>();
    preprocess(
        (int)opacities.size(0),
        (glm::vec3*)orig_points.contiguous().data_ptr<float>(),
        (shs_deg3_t*)shs.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        (cov3d_t*)cov3Ds.contiguous().data_ptr<float>(),
        width, height, block_x, block_y,
        glm::vec3({position_data[0], position_data[1], position_data[2]}), 
        glm::mat3({{rotation_data[0], rotation_data[1], rotation_data[2]},
                {rotation_data[3], rotation_data[4], rotation_data[5]},
                {rotation_data[6], rotation_data[7], rotation_data[8]}}),
        focal_x, focal_y, zFar, zNear,
        (float2*)points_xy.contiguous().data_ptr<float>(),
        (float4*)rgb_depth.contiguous().data_ptr<float>(),
        (float4*)conic_opacity.contiguous().data_ptr<float>(),
        (uint64_t*)gaussian_keys_unsorted.contiguous().data_ptr<int64_t>(), (uint32_t*)gaussian_values_unsorted.contiguous().data_ptr<int>(),
        curr_offset.data_ptr<int>());
}

void sort_gaussian_torch(int num_rendered,
    int width, int height, int block_x, int block_y,
	torch::Tensor& list_sorting_space,
	torch::Tensor& gaussian_keys_unsorted, torch::Tensor& gaussian_values_unsorted,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted)
{
    sort_gaussian(num_rendered,
        width, height, block_x, block_y,
        (char*)list_sorting_space.contiguous().data_ptr(), list_sorting_space.size(0),
        (uint64_t*)gaussian_keys_unsorted.contiguous().data_ptr<int64_t>(), (uint32_t*)gaussian_values_unsorted.contiguous().data_ptr<int>(),
        (uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(), (uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>());
}

void render_16x16_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
    torch::Tensor& ranges,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
    auto bg_color_data = bg_color.contiguous().data_ptr<float>();
    render_16x16(num_rendered,
        width, height,
        (float2*)points_xy.contiguous().data_ptr<float>(),
        (float4*)rgb_depth.contiguous().data_ptr<float>(),
        (float4*)conic_opacity.contiguous().data_ptr<float>(),
        (uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
        (uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
        (int2*)ranges.contiguous().data_ptr<int>(),
        float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
        (uchar3*)out_color.data_ptr());
}

void render_32x16_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
    torch::Tensor& ranges,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
    auto bg_color_data = bg_color.contiguous().data_ptr<float>();
    render_32x16(num_rendered,
        width, height,
        (float2*)points_xy.contiguous().data_ptr<float>(),
        (float4*)rgb_depth.contiguous().data_ptr<float>(),
        (float4*)conic_opacity.contiguous().data_ptr<float>(),
        (uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
        (uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
        (int2*)ranges.contiguous().data_ptr<int>(),
        float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
        (uchar3*)out_color.data_ptr());
}

void render_32x32_torch(int num_rendered,
	int width, int height,
	torch::Tensor& points_xy, torch::Tensor& rgb_depth, torch::Tensor& conic_opacity,
	torch::Tensor& gaussian_keys_sorted, torch::Tensor& gaussian_values_sorted,
    torch::Tensor& ranges,
	torch::Tensor& bg_color, torch::Tensor& out_color)
{
    auto bg_color_data = bg_color.contiguous().data_ptr<float>();
    render_32x32(num_rendered,
        width, height,
        (float2*)points_xy.contiguous().data_ptr<float>(),
        (float4*)rgb_depth.contiguous().data_ptr<float>(),
        (float4*)conic_opacity.contiguous().data_ptr<float>(),
        (uint64_t*)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
        (uint32_t*)gaussian_values_sorted.contiguous().data_ptr<int>(),
        (int2*)ranges.contiguous().data_ptr<int>(),
        float3{bg_color_data[0], bg_color_data[1], bg_color_data[2]},
        (uchar3*)out_color.data_ptr());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    auto ops = m.def_submodule("ops", "my custom operators");

    ops.def(
        "loadPly",
        &loadPly_torch,
        "load .ply file and return gaussian model data");

    ops.def(
        "preprocess",
        &preprocess_torch,
        "preprocess gaussian model data and generate key-value pairs");

    ops.def(
        "sort_gaussian",
        &sort_gaussian_torch,
        "sort gaussian key-value pairs");

    ops.def(
        "get_sort_buffer_size",
        &get_sort_buffer_size,
        "get sort buffer size");

    ops.def(
        "render_16x16",
        &render_16x16_torch,
        "sort key-value pairs and render");

    ops.def(
        "render_32x16",
        &render_32x16_torch,
        "sort key-value pairs and render");

    ops.def(
        "render_32x32",
        &render_32x32_torch,
        "sort key-value pairs and render");
}
