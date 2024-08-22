import torch
import flash_gaussian_splatting

import os
import sys
import json
import time
import cv2

class Scene:
    def __init__(self, device):
        self.device = device
        self.num_vertex = 0
        self.position = None
        self.shs = None
        self.opacity = None
        self.cov3d = None

    def loadPly(self, scene_path):
        self.num_vertex, self.position, self.shs, self.opacity, self.cov3d = flash_gaussian_splatting.ops.loadPly(
            scene_path)
        print("num_vertex = %d" % self.num_vertex)
        # 58*4byte
        self.position = self.position.to(self.device)  # 3
        self.shs = self.shs.to(self.device)  # 48
        self.opacity = self.opacity.to(self.device)  # 1
        self.cov3d = self.cov3d.to(self.device)  # 6


class Camera:
    def __init__(self, camera_json):
        self.id = camera_json['id']
        self.img_name = camera_json['img_name']
        self.width = camera_json['width']
        self.height = camera_json['height']
        self.position = torch.tensor(camera_json['position'])
        self.rotation = torch.tensor(camera_json['rotation'])
        self.focal_x = camera_json['fx']
        self.focal_y = camera_json['fy']
        self.zFar = 100.0
        self.zNear = 0.01


# 静态分配内存光栅化器
class Rasterizer:
    # 构造函数中分配内存
    def __init__(self, scene, MAX_NUM_RENDERED, SORT_BUFFER_SIZE, MAX_NUM_TILES):
        # 24byte
        self.gaussian_keys_unsorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int64)
        self.gaussian_values_unsorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int32)
        self.gaussian_keys_sorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int64)
        self.gaussian_values_sorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int32)

        self.list_sorting_space = torch.zeros(SORT_BUFFER_SIZE, device=scene.device, dtype=torch.int8)
        self.ranges = torch.zeros((MAX_NUM_TILES, 2), device=scene.device, dtype=torch.int32)
        self.curr_offset = torch.zeros(1, device=scene.device, dtype=torch.int32)

        # 9*4byte
        self.points_xy = torch.zeros((scene.num_vertex, 2), device=scene.device, dtype=torch.float32)
        self.depths = torch.zeros(scene.num_vertex, device=scene.device, dtype=torch.float32)
        self.rgb = torch.zeros((scene.num_vertex, 3), device=scene.device, dtype=torch.float32)
        self.conic_opacity = torch.zeros((scene.num_vertex, 4), device=scene.device, dtype=torch.float32)

    # 前向传播（应用层封装）
    def forward(self, scene, camera, bg_color):
        # 属性预处理 + 键值绑定
        self.curr_offset.fill_(0)
        flash_gaussian_splatting.ops.preprocess(scene.position, scene.shs, scene.opacity, scene.cov3d,
                                                camera.width, camera.height, 32, 16,
                                                camera.position, camera.rotation,
                                                camera.focal_x, camera.focal_y, camera.zFar, camera.zNear,
                                                self.points_xy, self.depths, self.rgb, self.conic_opacity,
                                                self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
                                                self.curr_offset)

        # 键值对数量判断 + 处理键值对过多的异常情况
        num_rendered = int(self.curr_offset.cpu()[0])
        # print(num_rendered)
        if num_rendered >= MAX_NUM_RENDERED:
            raise "Too many tils!"

        flash_gaussian_splatting.ops.sort_gaussian(num_rendered, camera.width, camera.height, 32, 16,
                                                   self.list_sorting_space,
                                                   self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
                                                   self.gaussian_keys_sorted, self.gaussian_values_sorted)
        # 排序 + 像素着色 + 混色阶段
        out_color = torch.zeros((camera.height, camera.width, 3), device=scene.device, dtype=torch.uint8)
        flash_gaussian_splatting.ops.render_32x16(num_rendered, camera.width, camera.height,
                                                  self.points_xy, self.depths, self.rgb, self.conic_opacity,
                                                  self.gaussian_keys_sorted, self.gaussian_values_sorted,
                                                  self.ranges, bg_color, out_color)
        return out_color

def save_png(image: torch.Tensor, path: str):    
    # image_bytes = image.cpu().numpy().tobytes()
    # image_uint8 = np.frombuffer(image_bytes, dtype=np.uint8).reshape(image.shape)    
    image_np = image.cpu().numpy() # .astype(np.uint8) 
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    # Save the image
    cv2.imwrite(path.replace("ppm", "jpeg"), image_bgr)

        
def savePpm(image, path):
    image = image.cpu()
    assert image.dim() >= 3
    assert image.size(2) == 3
    with open(path, 'wb') as f:
        f.write(b'P6\n' + f'{image.size(1)} {image.size(0)}\n255\n'.encode() + image.numpy().tobytes())


def render_scene(model_path, test_performance=False):
    scene_path = os.path.join(model_path, "point_cloud", "iteration_30000", "point_cloud.ply")
    print(scene_path)
    camera_path = os.path.join(model_path, "cameras.json")
    print(camera_path)
    device = torch.device('cuda:0')
    bg_color = torch.zeros(3, dtype=torch.float32)  # black

    scene = Scene(device)
    scene.loadPly(scene_path)

    with open(camera_path, 'r') as camera_file:
        cameras_json = json.loads(camera_file.read())

    image_dir = os.path.join(model_path, "test_out")
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    rasterizer = Rasterizer(scene, MAX_NUM_RENDERED, SORT_BUFFER_SIZE, MAX_NUM_TILES)
    for camera_json in cameras_json:
        camera = Camera(camera_json)
        print("image name = %s" % camera.img_name)

        image = rasterizer.forward(scene, camera, bg_color)  # warm up

        if test_performance:
            n = 10
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                image = rasterizer.forward(scene, camera, bg_color)  # test performance
            torch.cuda.synchronize()
            t1 = time.time()
            print("elapsed time = %f ms" % ((t1 - t0) / n * 1000))
            print("fps = %f" % (n / (t1 - t0)))

        image_path = os.path.join(image_dir, "%s.ppm" % camera.img_name)
        save_png(image, image_path)


MAX_NUM_RENDERED = 2 ** 27
SORT_BUFFER_SIZE = 2 ** 31
MAX_NUM_TILES = 2 ** 20

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        model_path = sys.argv[1]
        render_scene(model_path)
    else:
        models_path = ".\\results\\models"  # https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip
        for entry in os.scandir(models_path):
            if entry.is_dir():
                render_scene(entry.path)
