import time
import torch
import flash_gaussian_splatting


class Scene:
    def __init__(self, device):
        self.device = device
        self.num_vertex = 0
        self.position = None
        self.shs = None
        self.opacity = None
        self.cov3d = None

    def loadPly(self, scene_path):
        self.num_vertex, self.position, self.shs, self.opacity, self.cov3d = flash_gaussian_splatting.ops.loadPly(scene_path)
        print(self.num_vertex)
        # 58*4byte
        self.position = self.position.to(self.device) # 3
        self.shs = self.shs.to(self.device) # 48
        self.opacity = self.opacity.to(self.device) # 1
        self.cov3d = self.cov3d.to(self.device) # 6


class Camera:
    def __init__(self):
        self.width = 1957
        self.height = 1091
        self.position = torch.empty(3)
        self.position[0] = 3.3989622700470066
        self.position[1] = 0.6835340434263679
        self.position[2] = -2.2991544105562696
        self.rotation = torch.empty((3, 3))
        self.rotation[0, 0] = 0.7861880261556364
        self.rotation[0, 1] = -0.01644204414928749
        self.rotation[0, 2] = -0.6177686028875999
        self.rotation[1, 0] = -0.028784446228571188
        self.rotation[1, 1] = 0.9975867826238479
        self.rotation[1, 2] = -0.06318280453979654
        self.rotation[2, 0] = 0.6173166474223896
        self.rotation[2, 1] = 0.06745569151963769
        self.rotation[2, 2] = 0.783817508414292
        self.focal_x = 1163.2547280302354
        self.focal_y = 1156.2804049882861
        self.zFar = 100.0
        self.zNear = 0.01


# 静态分配内存光栅化器
class Rasterizer:
    # 构造函数中分配内存
    def __init__(self, scene, MAX_NUM_RENDERED, SORT_BUFFER_SIZE, MAX_NUM_TILES):
        # 24byte
        self.gaussian_keys_unsorted = torch.empty(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int64)
        self.gaussian_values_unsorted = torch.empty(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int32)
        self.gaussian_keys_sorted = torch.empty(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int64)
        self.gaussian_values_sorted = torch.empty(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int32)

        self.list_sorting_space = torch.empty(SORT_BUFFER_SIZE, device=scene.device, dtype=torch.int8)
        self.ranges = torch.empty((MAX_NUM_TILES, 2), device=scene.device, dtype=torch.int32)
        self.curr_offset = torch.empty(1, device=scene.device, dtype=torch.int32)

        # 9*4byte
        self.points_xy = torch.empty((scene.num_vertex, 2), device=scene.device, dtype=torch.float32)
        self.depths = torch.empty(scene.num_vertex, device=scene.device, dtype=torch.float32)
        self.rgb = torch.empty((scene.num_vertex, 3), device=scene.device, dtype=torch.float32)
        self.conic_opacity = torch.empty((scene.num_vertex, 4), device=scene.device, dtype=torch.float32)

    # 前向传播（应用层封装）
    def forward(self, scene, camera, bg_color):
        # 属性预处理 + 键值绑定
        flash_gaussian_splatting.ops.preprocess(scene.position, scene.shs, scene.opacity, scene.cov3d,
                                    camera.width, camera.height, 32, 16,
                                    camera.position, camera.rotation,
                                    camera.focal_x, camera.focal_y, camera.zFar, camera.zNear,
                                    self.points_xy, self.depths, self.rgb, self.conic_opacity,
                                    self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
                                    self.curr_offset)
        
        # 键值对数量判断 + 处理键值对过多的异常情况
        num_rendered = int(self.curr_offset.cpu()[0])
        print(num_rendered)
        if num_rendered >= MAX_NUM_RENDERED:
            raise

        flash_gaussian_splatting.ops.sort_gaussian(num_rendered, camera.width, camera.height, 32, 16,
                                            self.list_sorting_space,
                                            self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
                                            self.gaussian_keys_sorted, self.gaussian_values_sorted)
        # 排序 + 像素着色 + 混色阶段
        out_color = torch.zeros((camera.height, camera.width, 3), device=scene.device, dtype=torch.int8)
        flash_gaussian_splatting.ops.render_32x16(num_rendered, camera.width, camera.height,
                                self.points_xy, self.depths, self.rgb, self.conic_opacity,
                                self.gaussian_keys_sorted, self.gaussian_values_sorted,
                                self.ranges, bg_color, out_color)
        return out_color


def savePpm(image, path):
    image = image.cpu()
    assert image.dim() >= 3
    assert image.size(2) == 3
    with open(path, 'wb') as f:
        f.write(b'P6\n' + f'{image.size(1)} {image.size(0)}\n255\n'.encode() + image.numpy().tobytes())


if __name__ == "__main__":
    scene_path = "D:\\gaussian-splatting\\output\\truck_improve\\point_cloud\\iteration_30000\\point_cloud.ply"
    camera_path = "D:\\gaussian-splatting\\output\\truck_improve\\cameras.json"
    device = torch.device('cuda:0')
    bg_color = torch.zeros(3, dtype=torch.float32)  # black
    MAX_NUM_RENDERED = 2 ** 24
    SORT_BUFFER_SIZE = 2 ** 30
    MAX_NUM_TILES = 2 ** 20

    scene = Scene(device)
    scene.loadPly(scene_path)

    camera = Camera()

    rasterizer = Rasterizer(scene, MAX_NUM_RENDERED, SORT_BUFFER_SIZE, MAX_NUM_TILES)

    image = rasterizer.forward(scene, camera, bg_color)
    savePpm(image, "C:\\Users\\csy\\Desktop\\cuda_rasterizer\\000001.ppm")
