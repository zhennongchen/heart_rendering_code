"""Read the exported OBJ surfaces and render them with one fixed camera."""
from pathlib import Path
import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
SIZE = 1200
EPI_COLOR, ENDO_COLOR = "#18A99D", "#6840C6"
EPI_OPACITY, ENDO_OPACITY = 0.45, 0.82
CAMERA = [(3.2, -5.4, 2.2), (0, 0, 0), (0, 0, 1)]
PARALLEL_SCALE = 1.42
TRANSPARENT_BACKGROUND = False

meshes = {
    phase: {surface: pv.read(OUTPUT / f"{phase}_{surface}.obj")
            for surface in ("epi", "endo")}
    for phase in ("ed", "es")
}
# OBJ files retain unused vertices. Extract only referenced vertices for PCA.
points = np.concatenate([
    mesh.points[np.unique(mesh.faces.reshape(-1, 4)[:, 1:])]
    for mesh in meshes["ed"].values()
])
center = np.median(points, axis=0)
centered = points - center
values, vectors = np.linalg.eigh(centered.T @ centered)
order = np.argsort(values)[::-1]
long_axis, horizontal = vectors[:, order[0]].copy(), vectors[:, order[1]].copy()
side = np.cross(long_axis, horizontal)
side /= np.linalg.norm(side)
horizontal = np.cross(side, long_axis)
axes = np.column_stack([horizontal, side, long_axis])
provisional = centered @ axes
z, radius = provisional[:, 2], np.linalg.norm(provisional[:, :2], axis=1)
if np.median(radius[z <= np.percentile(z, 15)]) > np.median(radius[z >= np.percentile(z, 85)]):
    axes[:, 1:] *= -1
# PCA is a display heuristic, not an anatomical orientation annotation.
# Use ONE ED-based transform for both phases and surfaces to preserve size relations.
scale = np.ptp(centered @ axes, axis=0).max() / 2

for phase, surfaces in meshes.items():
    plotter = pv.Plotter(off_screen=True, window_size=(SIZE, SIZE))
    plotter.set_background("white")
    plotter.render_window.SetAlphaBitPlanes(1)
    plotter.render_window.SetMultiSamples(0)
    plotter.enable_anti_aliasing("ssaa")
    plotter.renderer.SetUseDepthPeeling(True)
    plotter.renderer.SetMaximumNumberOfPeels(16)
    plotter.renderer.SetOcclusionRatio(0.0)
    for surface, color, opacity in (
        ("endo", ENDO_COLOR, ENDO_OPACITY), ("epi", EPI_COLOR, EPI_OPACITY),
    ):
        mesh = surfaces[surface]
        mesh.points = (mesh.points.astype(np.float64) - center) @ axes / scale
        # Smooth shading changes normals, not geometry; do not fill the basal opening.
        plotter.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=True,
                         show_edges=False, ambient=0.30, diffuse=0.70,
                         specular=0.20, specular_power=18)
    plotter.add_light(pv.Light(position=(3.5, -4.5, 5.5), focal_point=(0, 0, 0),
                               color="white", intensity=0.85))
    plotter.camera_position = CAMERA
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = PARALLEL_SCALE
    plotter.reset_camera_clipping_range()
    # A hidden window still needs a working OpenGL graphics backend.
    plotter.screenshot(str(OUTPUT / f"heart_{phase}.png"),
                       transparent_background=TRANSPARENT_BACKGROUND)
    plotter.close()
    print("Saved", OUTPUT / f"heart_{phase}.png")
