import numpy as np
import mayavi
# import mayavi.mlab as mlab

def visualize_pts(pts, fig=None, bgcolor=(1, 1, 1), fgcolor=(0.0, 0.0, 0.0),
                  show_intensity=False, size=(600, 600), draw_origin=True):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    if show_intensity:
        # G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3:], mode='sphere',
                        #   colormap='gnuplot', scale_factor=1, figure=fig)
        rgba = np.zeros((pts.shape[0], 4), dtype=np.uint8)
        rgba[:, -1] = 255 # no transparency
        rgba[:, :3] = pts[:, 5:8] * 255
        print(rgba)
        pts = mlab.pipeline.scalar_scatter(pts[:, 0], pts[:, 1], pts[:, 2]) # plot the points
        pts.add_attribute(rgba, 'colors') # assign the colors to each point
        pts.data.point_data.set_active_scalars('colors')
        g = mlab.pipeline.glyph(pts)
        g.glyph.glyph.scale_factor = 0.07 # set scaling for all the points
        g.glyph.scale_mode = 'data_scaling_off' # make all the points same size
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)
    mlab.savefig('/home/ubuntu/Desktop/PointPainting-main/point5_paint.png')

    return fig


points = np.load('/home/ubuntu/Desktop/PointPainting-main/detector/data/kitti/training/painted_lidar/000005.npy')
#visualize_pts(points)




