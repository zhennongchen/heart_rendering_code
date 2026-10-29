"""Read the original VTK template, build graph tensors, and normalize positions."""
from pathlib import Path
import numpy as np
import pyvista as pv
import torch

ROOT = Path(__file__).resolve().parent
DATA, OUTPUT = ROOT / "data", ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

# Fixed similarity transform into the coordinate system used by this model.
# Recovered from the matching normalized template asset, not fitted to case 138.
# Using raw VTK coordinates directly with normalized displacements would be wrong.
SCALE = 0.017253034573270622
ROTATION = torch.tensor([
    [0.9452132671297325, -0.3263149187409748, -0.009510701814650647],
    [-0.32568134511082736, -0.9405747736590119, -0.09618085362014787],
    [0.02243972122684278, 0.09400887704557014, -0.9953184364553355],
], dtype=torch.float64)
TRANSLATION = torch.tensor([0.7353285987900233, -0.1682056757868186,
                            -0.04855029977377775], dtype=torch.float64)

# These are captured decoder outputs, not checkpoints. No model is invoked.
for phase in ("ed", "es"):
    displacement = torch.load(DATA / f"decoder_displacement_{phase}.pt",
                              map_location="cpu", weights_only=True)
    assert displacement.shape == (1, 22043, 3)
    print(phase.upper(), "decoder displacement:", tuple(displacement.shape))

# Read the ORIGINAL VTK. Do not clean, merge, sort or reindex its vertices.
mesh = pv.read(DATA / "LV_ED_mean.vtk")
pos = torch.from_numpy(np.asarray(mesh.points).copy()).float()
packed_faces = np.asarray(mesh.faces).reshape(-1, 4)
assert np.all(packed_faces[:, 0] == 3)
faces = torch.from_numpy(packed_faces[:, 1:].copy()).long()
assert pos.shape == (22043, 3) and faces.shape == (43840, 3)

# This specific original VTK stores 0=epi and 1=endo in point_data['scalars'].
# Verify that each triangle belongs to exactly one surface.
vertex_labels = torch.from_numpy(np.asarray(mesh.point_data["scalars"]).copy()).long()
assert set(vertex_labels.tolist()) == {0, 1}
assert torch.all(vertex_labels[faces] == vertex_labels[faces[:, :1]])
labels = vertex_labels[faces[:, 0]]

# Build both directions of every unique graph edge from the original faces.
edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
edges = torch.unique(edges.sort(dim=1).values, dim=0)
edge_index = torch.cat([edges, edges.flip(1)]).T.contiguous()
graph = {"pos": pos, "faces": faces, "edge_index": edge_index,
         "face_surface": labels, "vertex_surface": vertex_labels}
torch.save(graph, OUTPUT / "template_vtk_graph.pt")

# Transform only coordinates; the graph indices and connectivity remain unchanged.
normalized_pos = (SCALE * (pos.double() @ ROTATION) + TRANSLATION).float()
torch.save({**graph, "pos": normalized_pos}, OUTPUT / "template_graph.pt")
print("VTK -> graph tensors -> model coordinate system:", tuple(normalized_pos.shape))
