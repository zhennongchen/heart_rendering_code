"""Add decoder displacement tensors to template graph positions and export OBJ."""
from pathlib import Path
import numpy as np
import pyvista as pv
import torch

ROOT = Path(__file__).resolve().parent
DATA, OUTPUT = ROOT / "data", ROOT / "outputs"
graph = torch.load(OUTPUT / "template_graph.pt", map_location="cpu", weights_only=True)
with np.load(DATA / "world_transform.npz") as a:
    center, scale, orientation, axes, origin = [torch.as_tensor(a[k].copy()).double()
        for k in ("center", "scale", "orientation", "axes_world", "origin_world")]


def write_obj(path, vertices, faces):
    # VTK packs each triangle as [3, index0, index1, index2].
    vertices, faces = vertices.numpy(), faces.numpy()
    cells = np.column_stack([np.full(len(faces), 3), faces]).ravel()
    mesh = pv.PolyData(vertices, cells)
    mesh.save(path)  # The writer handles OBJ syntax and 1-based indices.


for phase in ("ed", "es"):
    displacement = torch.load(DATA / f"decoder_displacement_{phase}.pt",
                              map_location="cpu", weights_only=True)
    assert displacement.shape == (1, graph["pos"].shape[0], 3)

    # The central operation is Tensor + Tensor, with batch size exactly 1.
    # [1, N, 3] = [1, N, 3] + [1, N, 3]. No inference or matching is performed.
    deformed = graph["pos"].unsqueeze(0) + displacement

    # Only vertex coordinates change. Faces, edges and labels stay the same.
    deformed_graph = {**graph, "pos": deformed}
    torch.save(deformed_graph, OUTPUT / f"{phase}_deformed_graph.pt")

    # Restore physical coordinates with the saved case transform.
    world = (((deformed.double() * scale + center) @ orientation) @ axes + origin).float()
    torch.save({**deformed_graph, "pos": world}, OUTPUT / f"{phase}_world_graph.pt")
    faces, labels = graph["faces"], graph["face_surface"]
    write_obj(OUTPUT / f"{phase}_shell.obj", world[0], faces)
    write_obj(OUTPUT / f"{phase}_epi.obj", world[0], faces[labels == 0])
    write_obj(OUTPUT / f"{phase}_endo.obj", world[0], faces[labels == 1])
    print(phase.upper(), "deformed graph positions:", tuple(deformed.shape), "-> OBJ")
