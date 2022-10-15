import matplotlib.pyplot as plt
from mvtorch.mvrenderer import MVRenderer

class PointCloudVisualizer():
    
    def __init__(self, nb_views, image_size=224, object_color=(1, 1, 1), background_color=(0, 0, 0), points_radius=0.006, points_per_pixel=1):
        self.nb_views = nb_views
        self.image_size = image_size
        self.object_color = object_color
        self.background_color = background_color
        self.points_radius = points_radius
        self.points_per_pixel = points_per_pixel

        self.renderer = MVRenderer(
            nb_views=self.nb_views,
            image_size=self.image_size,
            pc_rendering=True,
            object_color=self.object_color,
            background_color=self.background_color,
            points_per_pixel=self.points_per_pixel,
            return_mapping=False
        )

    def visualize_inline(self, points, dist, elev, azim):
        
        rendered_images, _ = self.renderer(None, points, azim=azim, elev=elev, dist=dist, color=self.object_color)

        batch_size = rendered_images.shape[0]
        nb_views = rendered_images.shape[1]
        
        rendered_images = rendered_images.cpu()

        fig, axs = plt.subplots(nrows=batch_size, ncols=nb_views, squeeze=False, figsize=(4*nb_views, 4*batch_size))
        for i in range(batch_size):
            for j in range(nb_views):
                axs[i, j].imshow(rendered_images[i, j].permute(1, 2, 0))
                axs[i, j].axis("off")
        fig.tight_layout()