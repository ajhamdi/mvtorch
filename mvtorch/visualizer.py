import torch
import matplotlib.pyplot as plt
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    NormWeightedCompositor
)

class PointCloudVisualizer():
    
    def __init__(self, image_size=512, object_color=(1, 1, 1), background_color=(0, 0, 0), radius=0.006, points_per_pixel=1):
        self.image_size = image_size
        self.object_color = torch.Tensor(object_color)
        self.background_color = torch.Tensor(background_color)
        self.radius = radius
        self.points_per_pixel = points_per_pixel
        self.raster_settings = PointsRasterizationSettings(
            image_size = self.image_size, 
            radius = self.radius,
            points_per_pixel = self.points_per_pixel,
        )

    def visualize_offline(self, point_cloud, dist, elev, azim):
        R, T = look_at_view_transform(dist, elev, azim)
        cameras = FoVOrthographicCameras(R=R, T=T, znear=0.01)

        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=NormWeightedCompositor(background_color=self.background_color)
        )

        point_cloud = Pointclouds(points=[point_cloud], features=[(self.object_color * torch.ones_like(point_cloud))])
        images = renderer(point_cloud)
        plt.figure(figsize=(10, 10))
        plt.imshow(images[0, ..., :3].cpu().numpy())
        plt.axis("off")