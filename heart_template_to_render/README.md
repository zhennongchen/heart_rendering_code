# Decoder 位移 Tensor → template 图 Tensor → 相加 → OBJ → 图片

本例从病例 **138** 已截取的 decoder 位移 **PyTorch Tensor** 开始，batch=1。

## 完整数据流

```text
data/LV_ED_mean.vtk                         已截取的 decoder 位移
原始 template                              data/decoder_displacement_ed.pt
        │                                  data/decoder_displacement_es.pt
        │ 01_model_output.py                       │
        ↓                                          │ torch.load()
读取顶点、三角面、内外膜标签                        ↓
转为图 Tensor，按三角面构造双向边             每阶段 displacement
        │                                   Float Tensor [1,22043,3]
        ├→ outputs/template_vtk_graph.pt            │
        │  保留原始 VTK 坐标                       │
        ↓                                          │
固定相似变换：缩放、旋转、平移                      │
        ↓                                          │
outputs/template_graph.pt                          │
pos          [22043,3]，归一化坐标                  │
faces        [43840,3]                              │
edge_index   [2,131762]                             │
内外膜标签                                         │
        │                                          │
        └──────────────────┬───────────────────────┘
                           │ 02_apply_template.py
                           ↓
      deformed = graph["pos"].unsqueeze(0) + displacement
                  [1,22043,3] + [1,22043,3]
                           ↓
      outputs/ed_deformed_graph.pt / es_deformed_graph.pt
      pos 为变形后的归一化坐标 [1,22043,3]
      faces、edge_index、内外膜标签保持不变
                           │
                           │ data/world_transform.npz
                           ↓
      恢复世界坐标：scale、center、orientation、axes、origin
                           ↓
      outputs/ed_world_graph.pt / es_world_graph.pt
      pos 为世界坐标 Tensor [1,22043,3]
                           ↓
      取出唯一病例 world[0] → Tensor 转 NumPy
      vertices [22043,3] + faces [M,3]
                           ↓
      pv.PolyData(vertices, cells) → mesh.save(...)
                           ↓
      outputs/ed_shell.obj、ed_epi.obj、ed_endo.obj
      outputs/es_shell.obj、es_epi.obj、es_endo.obj
                           │ 03_render.py
                           ↓
      pv.read() 读取内外膜 OBJ → 共用展示坐标变换
                           ↓
      设置内外膜材质、透明度、灯光 → 固定正交相机 → 截图
                           ↓
      outputs/heart_ed.png / heart_es.png
```

ED、ES 分别处理，每阶段 batch 都是 1；运行过程中不调用模型。
顶点对应关系来自原模型与 template 保持一致的索引顺序，后处理按同一索引相加，
不会根据坐标重新匹配顶点。原始 VTK 坐标、模型归一化坐标、病例世界坐标和渲染展示坐标
是不同阶段的坐标空间，只有进入同一模型坐标空间后才能相加。

## 输入文件

```text
heart_template_to_render_case138/
├── 01_model_output.py        # 读取原始 VTK，图 Tensor 化，并变换到模型坐标
├── 02_apply_template.py      # 两个 Tensor 相加，转世界坐标，导出 OBJ
├── 03_render.py              # 读取 OBJ，固定机位渲染
├── requirements.txt
├── data/
│   ├── decoder_displacement_ed.pt  # ED 位移 Tensor [1,22043,3]
│   ├── decoder_displacement_es.pt  # ES 位移 Tensor [1,22043,3]
│   ├── LV_ED_mean.vtk               # 原始 template，完整保留原文件
│   └── world_transform.npz         # 坐标恢复参数
└── outputs/                        # 图 Tensor、OBJ 和图片
```

两个 `.pt` 文件分别用 `torch.load(..., weights_only=True)` 读取，得到
`torch.float32` Tensor `[1,22043,3]`，不是模型权重。
两个文件代表同一病例的 ED、ES，**不是两个 batch 样本**。
位移是 decoder 已限幅、尚未加 template 的归一化 XYZ 位移。

数据来自此前已完成的真实推理。本次只将已截取数据按阶段保存为 Tensor 文件，
保留 batch 维度并核对数值逐位一致，没有重新运行模型。

## 本地运行

复制整个文件夹到本地，在该目录打开终端：

```bash
python -m pip install -r requirements.txt
python 01_model_output.py
python 02_apply_template.py
python 03_render.py
```

Tensor 操作默认在 CPU 上完成，不需要 CUDA。
渲染需要本机正常工作的 OpenGL 和图形驱动。

## 第 01 步：读取原始 VTK → 图 Tensor → 模型坐标

代码实际从 `pv.read(DATA / "LV_ED_mean.vtk")` 开始，读取原始顶点及三角面。
该 VTK 的点属性 `scalars` 保存内外膜标签：0=外膜、1=内膜。
不排序、不合并、不重新编号顶点。

template 的图用普通 Python 字典保存，无需额外安装图学习库：

| 字段 | 类型与形状 | 含义 |
|---|---|---|
| `pos` | Float Tensor `[22043,3]` | 顶点坐标 |
| `faces` | Long Tensor `[43840,3]` | 三角面顶点索引 |
| `edge_index` | Long Tensor `[2,E]` | 从三角面提取的双向边，E 为有向边数 |
| `face_surface` | Long Tensor `[43840]` | 0=外膜，1=内膜 |
| `vertex_surface` | Long Tensor `[22043]` | 每个顶点的内外膜标签 |

代码将每个三角面拆成三条边，去重后保存两个方向。
原始坐标的图首先写入 `outputs/template_vtk_graph.pt`。

原始 VTK 坐标不能直接加归一化位移，必须先使用固定相似变换：

```python
normalized_pos = (SCALE * (pos.double() @ ROTATION) + TRANSLATION).float()
```

`SCALE`、`ROTATION`、`TRANSLATION` 已直接写在第 01 个脚本顶部。
原变换报告未找到，因此这些参数从原始 VTK 与现有模型归一化 template 资产恢复，
不是对病例 138 重新拟合。与模型资产的最大归一化坐标残差为约 `5.96×10⁻⁸`，
来源与误差记录在 `PROVENANCE.json`。正常运行不读取任何 template NPZ。

变换后的图写入 `outputs/template_graph.pt`，仅改变坐标，顶点索引和连接关系保持不变。

## 第 02 步：Tensor 相加就是施加位移

```python
# template 顶点：[N,3]
# decoder 位移：[1,N,3]
deformed = graph["pos"].unsqueeze(0) + displacement
# deformed：[1,N,3]
```

逐点对应关系是：

```python
deformed[0, i, :] = graph["pos"][i, :] + displacement[0, i, :]
```

位移与 template 必须共享顶点顺序、归一化坐标空间及尺度。
只更新图的 `pos`，`faces`、`edge_index` 和标签全部保持。
图连接关系无需参与加法，但必须随变形后的顶点保留。

输出每阶段的归一化图：`ed_deformed_graph.pt`、`es_deformed_graph.pt`。
它们的 `pos` 均为 `[1,22043,3]`。

接着仍用 PyTorch Tensor 运算恢复世界坐标：

```python
world = ((deformed * scale + center) @ orientation) @ axes + origin
```

参数来自 `world_transform.npz`。世界坐标图保存为
`ed_world_graph.pt`、`es_world_graph.pt`。
**到导出 OBJ 时才将 Tensor 转为 NumPy 数组。**

通过 PyVista 构造 mesh 并保存，无需自行逐行写 OBJ：

```python
vertices, faces = vertices.numpy(), faces.numpy()
cells = np.column_stack([np.full(len(faces), 3), faces]).ravel()
mesh = pv.PolyData(vertices, cells)
mesh.save(path)
```

`cells` 中每个三角面是 `[3, 顶点索引0, 顶点索引1, 顶点索引2]`。

导出结果：

- `ed_shell.obj`、`es_shell.obj`：完整内外膜。
- `ed_epi.obj`、`es_epi.obj`：外膜。
- `ed_endo.obj`、`es_endo.obj`：内膜。

OBJ 索引从 1 开始，Tensor 索引从 0 开始，由 PyVista/VTK 保存函数自动转换。
这里没有重新做图编码、平滑、删面或补洞。

## 第 03 步：OBJ → 固定机位图片

```text
读取 epi.obj / endo.obj
    → 共用展示变换
    → PyVista/VTK 设置材质和灯光
    → 固定正交相机
    → heart_ed.png / heart_es.png
```

颜色、透明度、机位和分辨率均在 `03_render.py` 顶部。
两阶段共用 ED 拟合的展示变换；ES 不会单独放大。
PCA 只是展示方向启发式，不是解剖标注；展示变换不修改原 OBJ。
透明叠加细节可能因显卡、驱动和渲染后端而略有不同。

## 核对

`PROVENANCE.json` 记录截取数据来源及输入 SHA256。
`VALIDATION.json` 记录 batch=1、Tensor 相加、图连接保持及导出结果的验证。
原模型最终返回的顶点本身已含 template 加法；本例特意从这一加法**之前的位移**开始。
