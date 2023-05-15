import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from ipywidgets import widgets
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from typing import List, Tuple, Dict, Set, Iterable, Callable
from spine_fitter import SpineGrouping
from spine_segmentation import point_2_list, list_2_point, hash_point, \
    Segmentation, segmentation_by_distance, local_threshold_3d,\
    spines_to_segmentation, correct_segmentation, get_spine_meshes, apply_scale
import meshplot as mp
from IPython.display import display
from spine_metrics import SpineMetric, SpineMetricDataset, calculate_metrics, \
    HistogramSpineMetric, FloatSpineMetric, MeshDataset, LineSet, \
    OldChordDistributionSpineMetric
from scipy.ndimage import measurements
from spine_clusterization import SpineClusterizer, KMeansSpineClusterizer, DBSCANSpineClusterizer
from pathlib import Path
import os
from sklearn.linear_model import LinearRegression
from functools import cmp_to_key
from spine_clusterization import ks_test
from CGAL.CGAL_Polygon_mesh_processing import Polylines
from CGAL.CGAL_Surface_mesh_skeletonization import surface_mesh_skeletonization
from scipy.spatial.distance import euclidean
import csv


Color = Tuple[float, float, float]

RED = (1, 0, 0)
GREEN = (0, 1, 0)
BLUE = (0, 0, 1)
WHITE = (1, 1, 1)
YELLOW = (1, 0.8, 0)
GRAY = (0.69, 0.69, 0.69)
DARK_GRAY = (0.30, 0.30, 0.30)
BLACK = (0.0, 0.0, 0.0)

V_F = Tuple[np.ndarray, np.ndarray]


class SpineMeshDataset:
    # spine name -> Polyhedron_3
    spine_meshes: MeshDataset
    # dendrite name -> Polyhedron_3
    dendrite_meshes: MeshDataset
    spine_to_dendrite: Dict[str, str]
    dendrite_to_spines: Dict[str, Set[str]]
    # spine name -> v f
    spine_v_f: Dict[str, V_F]
    # dendrite name -> v f
    dendrite_v_f: Dict[str, V_F]

    def __init__(self, spine_meshes: MeshDataset = None, dendrite_meshes: MeshDataset = None,
                 spine_to_dendrite: Dict[str, str] = None) -> None:
        if spine_meshes is None:
            spine_meshes = {}
        if dendrite_meshes is None:
            dendrite_meshes = {}
        if spine_to_dendrite is None:
            spine_to_dendrite = {}
            
        # set fields
        self.spine_meshes = spine_meshes
        self.dendrite_meshes = dendrite_meshes
        self.spine_to_dendrite = spine_to_dendrite

        # generate mapping of dendrites to their spines
        self.dendrite_to_spines = {name: set() for name in dendrite_meshes.keys()}
        for (spine_name, dendrite_name) in spine_to_dendrite.items():
            self.dendrite_to_spines[dendrite_name].add(spine_name)

        # calculate 'meshplot' mesh representations
        self._calculate_v_f()

    @property
    def spine_names(self) -> Set[str]:
        return set(self.spine_meshes.keys())

    @property
    def dendrite_names(self) -> Set[str]:
        return set(self.dendrite_meshes.keys())

    def get_dendrite_mesh(self, spine_name: str) -> Polyhedron_3:
        return self.dendrite_meshes[self.spine_to_dendrite[spine_name]]

    def get_dendrite_v_f(self, spine_name: str) -> V_F:
        return self.dendrite_v_f[self.spine_to_dendrite[spine_name]]

    def apply_scale(self, scale: Color) -> None:
        def _apply_scale(mesh_dataset: MeshDataset) -> None:
            for (name, mesh) in mesh_dataset.items():
                mesh_dataset[name] = apply_scale(mesh, scale)
        _apply_scale(self.spine_meshes)
        _apply_scale(self.dendrite_meshes)
        self._calculate_v_f()

    def load(self, folder_path: str = "output",
             spine_file_pattern: str = "**/spine_*.off") -> "SpineMeshDataset":
        spine_meshes = {}
        dendrite_meshes = {}
        spine_to_dendrite = {}
        path = Path(folder_path)
        spine_names = list(path.glob(spine_file_pattern))
        for spine_name in spine_names:
            spine_meshes[str(spine_name)] = Polyhedron_3(str(spine_name))
            dendrite_path = str(spine_name.parent) + "\\surface_mesh.off"
            if dendrite_path not in dendrite_meshes:
                dendrite_meshes[dendrite_path] = Polyhedron_3(dendrite_path)
            spine_to_dendrite[str(spine_name)] = dendrite_path
        self.__init__(spine_meshes, dendrite_meshes, spine_to_dendrite)
        return self

    def _calculate_v_f(self) -> None:
        self.spine_v_f = preprocess_meshes(self.spine_meshes)
        self.dendrite_v_f = preprocess_meshes(self.dendrite_meshes)


def _mesh_to_v_f(mesh: Polyhedron_3) -> V_F:
    vertices = np.ndarray((mesh.size_of_vertices(), 3))
    for i, vertex in enumerate(mesh.vertices()):
        vertex.set_id(i)
        vertices[i, :] = point_2_list(vertex.point())

    facets = np.ndarray((mesh.size_of_facets(), 3)).astype("uint")
    for i, facet in enumerate(mesh.facets()):
        circulator = facet.facet_begin()
        j = 0
        begin = facet.facet_begin()
        while circulator.hasNext():
            halfedge = circulator.next()
            v = halfedge.vertex()
            facets[i, j] = (v.id())
            j += 1
            # check for end of loop
            if circulator == begin or j == 3:
                break
    return vertices, facets


def create_dir(dir_name: str) -> None:
    try:
        os.mkdir(dir_name)
    except OSError as _:
        pass


def remove_file(file_path: str) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)


def preprocess_meshes(spine_meshes: MeshDataset) -> Dict[str, V_F]:
    output = {}
    for (spine_name, spine_mesh) in spine_meshes.items():
        output[spine_name] = _mesh_to_v_f(spine_mesh)
    return output


def show_3d_mesh(mesh: Polyhedron_3, scale: Color = (1, 1, 1)) -> None:
    shown_mesh = apply_scale(mesh, scale)
    v, f = _mesh_to_v_f(shown_mesh)
    mp.plot(v, f)


def polylines_to_line_set(polylines: Polylines) -> LineSet:
    output = []
    for line in polylines:
        for i in range(len(line) - 1):
            output.append((line[i], line[i + 1]))
    return output


def show_polylines(polylines: Polylines, mesh: Polyhedron_3 = None) -> None:
    show_line_set(polylines_to_line_set(polylines), mesh)


def _add_line_set_to_viewer(viewer: mp.Viewer, lines: LineSet) -> None:
    viewer.add_lines(np.array([point_2_list(line[0]) for line in lines]),
                     np.array([point_2_list(line[1]) for line in lines]),
                     shading={"line_color": "red"})


def _add_mesh_to_viewer_as_wireframe(viewer: mp.Viewer, mesh_v_f: V_F) -> None:
    (v, f) = mesh_v_f
    starts = []
    ends = []
    for facet in f:
        starts.append(v[facet[0]])
        starts.append(v[facet[1]])
        starts.append(v[facet[2]])
        ends.append(v[facet[1]])
        ends.append(v[facet[2]])
        ends.append(v[facet[0]])
    viewer.add_lines(np.array(starts), np.array(ends),
                     shading={"line_color": "gray"})


def show_line_set(lines: LineSet, mesh: Polyhedron_3 = None) -> None:
    view = mp.Viewer({})
    _add_line_set_to_viewer(view, lines)
    if mesh:
        _add_mesh_to_viewer_as_wireframe(view, _mesh_to_v_f(mesh))
    display(view._renderer)


def _show_image(ax, image, mask=None, mask_opacity=0.5,
                cmap="gray", title=None):
    if mask is not None:
        indices = mask > 0
        mask = np.stack([np.zeros_like(mask), mask, np.zeros_like(mask)], -1)
        image = np.stack([image, image, image], -1)
        image[indices] = image[indices] * (1 - mask_opacity) + mask[indices] * mask_opacity

    ax.imshow(image, norm=Normalize(0, 255), cmap=cmap)

    if title:
        ax.set_title(title)


def _show_cross_planes(ax, coord_1, coord_2, shape, color_1, color_2, border_color) -> None:
    # show plane 1
    ax.plot((coord_1, coord_1), (0, shape[0] - 1), color=color_1, lw=3)
    # show plane 2
    ax.plot((0, shape[1] - 1), (coord_2, coord_2), color=color_2, lw=3)
    # show border
    # horizontal
    ax.plot((0, shape[1] - 1), (0, 0), color=border_color, lw=3)
    ax.plot((0, shape[1] - 1), (shape[0] - 1, shape[0] - 1), color=border_color, lw=3)
    # vertical
    ax.plot((0, 0), (0, shape[0] - 1), color=border_color, lw=3)
    ax.plot((shape[1] - 1, shape[1] - 1), (0, shape[0] - 1), color=border_color, lw=3)


def make_viewer(width: int = 600, height: int = 600) -> mp.Viewer:
    return mp.Viewer({"width": width, "height": height})


def make_colors(v_f: V_F, color: Color) -> np.ndarray:
    num_of_v = v_f[0].shape[0]
    colors = np.ndarray((num_of_v, 3))
    for i in range(num_of_v):
        colors[i] = color
    return colors


def _segmentation_to_colors(vertices: np.ndarray,
                            segmentation: Segmentation,
                            dendrite_color: Color = GREEN,
                            spine_color: Color = RED) -> np.ndarray:
    colors = np.ndarray((vertices.shape[0], 3))
    for i, vertex in enumerate(vertices):
        if hash_point(list_2_point(vertex)) in segmentation:
            colors[i] = spine_color
        else:
            colors[i] = dendrite_color
    return colors


def _grouping_to_colors(vertices: np.ndarray,
                        spine_meshes: MeshDataset,
                        grouping: SpineGrouping,
                        dendrite_color: Color = GREEN) -> np.ndarray:
    # generate segmentation for each group
    group_segmentations = []
    outlier_group = grouping.outlier_group
    group_colors = []
    for (label, group) in grouping.groups.items():
        group_segmentations.append(spines_to_segmentation([spine_meshes[name] for name in group]))
        group_colors.append(grouping.colors[label])
    group_segmentations.append(spines_to_segmentation([spine_meshes[name] for name in outlier_group]))
    group_colors.append(BLACK)

    # for each vertex check if belongs to group
    colors = np.ndarray((vertices.shape[0], 3))
    for i, vertex in enumerate(vertices):
        hp = hash_point(list_2_point(vertex))
        colors[i] = dendrite_color
        for segmentation, color in zip(group_segmentations, group_colors):
            if hp in segmentation:
                colors[i] = color[:3]
                break
    return colors


class SpinePreview:
    widget: widgets.Widget
    spine_viewer: mp.Viewer
    dendrite_viewer: mp.Viewer

    spine_mesh: Polyhedron_3
    spine_name: str
    metrics: List[SpineMetric]

    _spine_color: Color

    _spine_colors: np.ndarray

    _dendrite_colors: np.ndarray

    _spine_v_f: V_F
    _dendrite_v_f: V_F

    _metrics_box: widgets.VBox

    _spine_mesh_id: int

    def __init__(self, spine_mesh: Polyhedron_3,
                 spine_v_f: V_F,
                 dendrite_v_f: V_F,
                 metrics: List[SpineMetric],
                 spine_name: str,
                 spine_color: Color = RED) -> None:
        self._spine_color = spine_color
        self._spine_mesh_id = 0
        self._dendrite_v_f = dendrite_v_f
        self._set_spine_mesh(spine_mesh, spine_v_f, metrics)
        self.spine_name = spine_name
        self.create_views()

    @property
    def spine_color(self) -> Color:
        return self._spine_color

    @spine_color.setter
    def spine_color(self, new_spine_color: Color) -> None:
        self._spine_color = new_spine_color
        self._make_colors()
        self.spine_viewer.update_object(colors=self._get_spine_colors())
        self.dendrite_viewer.update_object(colors=self._get_dendrite_colors())

    def create_views(self) -> None:
        preview_panel = widgets.HBox(children=[self._make_dendrite_view(),
                                               self._make_spine_panel()],
                                     layout=widgets.Layout(align_items="flex-start"))
        self.widget = widgets.VBox([widgets.Label(self.spine_name), preview_panel])

    def _set_spine_mesh(self, spine_mesh: Polyhedron_3, spine_v_f, metrics: List[SpineMetric]) -> None:
        self.spine_mesh = spine_mesh
        self._spine_v_f = spine_v_f

        self._make_colors()

        if hasattr(self, "spine_viewer"):
            # self.spine_viewer.update_object(vertices=self._spine_v_f[0],
            #                                 faces=self._spine_v_f[1],
            #                                 colors=self._get_spine_colors())
            # self.spine_viewer.update_object(colors=self._get_spine_colors())
            self.spine_viewer.remove_object(self._spine_mesh_id)
            self._spine_mesh_id += 1
            self.spine_viewer.add_mesh(*self._spine_v_f, self._get_spine_colors())
        if hasattr(self, "dendrite_viewer"):
            self.dendrite_viewer.update_object(colors=self._get_dendrite_colors())

        self.metrics = metrics
        if hasattr(self, "_metrics_box"):
            self._fill_metrics_box()

    def _make_colors(self) -> None:
        # dendrite view colors
        self._dendrite_colors = np.ndarray((len(self._dendrite_v_f[0]), 3))
        self._dendrite_colors[:] = \
            _segmentation_to_colors(self._dendrite_v_f[0],
                                    spines_to_segmentation([self.spine_mesh]),
                                    GRAY, self.spine_color)
        # spine view colors
        self._spine_colors = np.ndarray((self.spine_mesh.size_of_vertices(), 3))
        self._spine_colors[:] = self.spine_color

    def _get_dendrite_colors(self):
        return self._dendrite_colors

    def _get_spine_colors(self):
        return self._spine_colors

    def _make_dendrite_view(self) -> widgets.Widget:
        # make mesh viewer
        self.dendrite_viewer = make_viewer(400, 600)
        self.dendrite_viewer.add_mesh(*self._dendrite_v_f, self._get_dendrite_colors())

        # set layout
        self.dendrite_viewer._renderer.layout = widgets.Layout(border="solid 1px")

        # title
        title = widgets.Label("Full View")

        return widgets.VBox(children=[title, self.dendrite_viewer._renderer])

    def _make_spine_view(self) -> widgets.Widget:
        # make mesh viewer
        self.spine_viewer = make_viewer(200, 200)
        self.spine_viewer.add_mesh(*self._spine_v_f, self._get_spine_colors())

        # set layout
        self.spine_viewer._renderer.layout = widgets.Layout(border="solid 1px")

        # title
        title = widgets.Label("Spine View")

        return widgets.VBox(children=[title, self.spine_viewer._renderer])

    def _fill_metrics_box(self) -> None:
        self._metrics_box.children = [widgets.VBox([widgets.Label(metric.name),
                                                    metric.show()],
                                                   layout=widgets.Layout(border="solid 1px"))
                                      for metric in self.metrics]

    def _make_metrics_panel(self) -> widgets.Widget:
        self._metrics_box = widgets.VBox([], layout=widgets.Layout())
        self._fill_metrics_box()

        return widgets.VBox(children=[widgets.Label("Metrics"), self._metrics_box])

    def _make_spine_panel(self) -> widgets.Widget:
        # convert spine mesh to meshplot format
        return widgets.VBox(children=[self._make_spine_view(),
                                      self._make_metrics_panel()],
                            layout=widgets.Layout(align_items="flex-start"))


class SelectableSpinePreview(SpinePreview):
    is_selected_checkbox: widgets.Checkbox
    is_selected: bool
    on_selected: Callable[["SelectableSpinePreview"], None] = None

    _unselected_spine_colors: np.ndarray
    _unselected_dendrite_colors: np.ndarray

    metrics: List[SpineMetric]
    _metric_names: List[str]
    _metric_params: List[Dict]

    _correction_slider: widgets.IntSlider

    _dendrite_mesh: Polyhedron_3

    _initial_segmentation: Segmentation

    def __init__(self, spine_mesh: Polyhedron_3,
                 spine_v_f: V_F,
                 dendrite_v_f: V_F,
                 dendrite_mesh: Polyhedron_3,
                 metric_names: List[str],
                 metric_params: List[Dict] = None) -> None:
        self._initial_segmentation = spines_to_segmentation([spine_mesh])
        self.is_selected = True
        self._make_is_selected()
        self._dendrite_mesh = dendrite_mesh
        self._make_correction_slider()
        self._metric_names = metric_names
        self._metric_params = metric_params
        self._metrics = calculate_metrics(spine_mesh, self._metric_names,
                                          self._metric_params)
        super().__init__(spine_mesh, spine_v_f, dendrite_v_f, self._metrics, "")

    def create_views(self) -> None:
        super().create_views()
        self.widget = widgets.VBox([self.is_selected_checkbox,
                                    self._correction_slider,
                                    self.widget])

    def _make_correction_slider(self) -> None:
        def observe_correction(change: Dict) -> None:
            if change["name"] == "value":
                self._correct_spine(change["new"])
        self._correction_slider = widgets.IntSlider(min=-6, max=6, value=0,
                                                    continuous_update=False,
                                                    description="Correction")
        self._correction_slider.observe(observe_correction)

    def _correct_spine(self, correction_value: int) -> None:
        new_segm = correct_segmentation(self._initial_segmentation,
                                        self._dendrite_mesh, correction_value)
        meshes = get_spine_meshes(self._dendrite_mesh, new_segm)
        if len(meshes) != 1:
            print(f"Oops, split this spine into {len(meshes)} spines.")
        self._set_spine_mesh(meshes[0], _mesh_to_v_f(meshes[0]),
                             calculate_metrics(meshes[0], self._metric_names, self._metric_params))

    def _make_is_selected(self) -> None:
        def update_is_selected(change: Dict) -> None:
            if change["name"] == "value":
                self.set_selected(change["new"])
        self.is_selected_checkbox = widgets.Checkbox(value=self.is_selected,
                                                     description="Valid spine")
        self.is_selected_checkbox.observe(update_is_selected)

    def _make_colors(self) -> None:
        super()._make_colors()
        # dendrite view colors
        self._unselected_dendrite_colors = np.ndarray(
            (len(self._dendrite_v_f[0]), 3))
        self._unselected_dendrite_colors[:] = \
            _segmentation_to_colors(self._dendrite_v_f[0],
                                    spines_to_segmentation([self.spine_mesh]),
                                    GRAY, DARK_GRAY)
        # spine view colors
        self._unselected_spine_colors = np.ndarray(
            (self.spine_mesh.size_of_vertices(), 3))
        self._unselected_spine_colors[:] = GRAY

    def _get_dendrite_colors(self):
        if self.is_selected:
            return self._dendrite_colors
        return self._unselected_dendrite_colors

    def _get_spine_colors(self):
        if self.is_selected:
            return self._spine_colors
        return self._unselected_spine_colors

    def set_selected(self, value: bool) -> None:
        self.is_selected = value
        self.spine_viewer.update_object(self._spine_mesh_id, colors=self._get_spine_colors())
        self.dendrite_viewer.update_object(colors=self._get_dendrite_colors())
        if self.on_selected is not None:
            self.on_selected(self)


def _make_navigation_widget(slider: widgets.IntSlider, step=1) -> widgets.Widget:
    next_button = widgets.Button(description=">")
    prev_button = widgets.Button(description="<")

    def disable_buttons(_=None) -> None:
        next_button.disabled = slider.value >= slider.max
        prev_button.disabled = slider.value <= slider.min
        
    disable_buttons()
    slider.observe(disable_buttons)

    def next_callback(_: widgets.Button) -> None:
        slider.value += step
        disable_buttons()

    def prev_callback(_: widgets.Button) -> None:
        slider.value -= step
        disable_buttons()

    next_button.on_click(next_callback)
    prev_button.on_click(prev_callback)

    box = widgets.HBox([prev_button, next_button])
    return box
    

def select_spines_widget(spine_meshes: List[Polyhedron_3],
                         dendrite_mesh: Polyhedron_3,
                         metric_names: List[str],
                         metric_params: List[Dict] = None) -> widgets.Widget:
    # create selectable 3d previews for each spine
    dendrite_v_f: V_F = _mesh_to_v_f(dendrite_mesh)
    spine_previews = [SelectableSpinePreview(spine_mesh, _mesh_to_v_f(spine_mesh),
                                             dendrite_v_f, dendrite_mesh, metric_names,
                                             metric_params)
                      for spine_mesh in spine_meshes]

    # set callbacks to update selected spines on checkbox toggle
    selection = []

    def on_selected_callback(_: SelectableSpinePreview) -> None:
        selection.clear()
        selection.extend([(selected_preview.spine_mesh, selected_preview.metrics)
                          for selected_preview in spine_previews if selected_preview.is_selected])

    for preview in spine_previews:
        preview.on_selected = on_selected_callback

    # show previews
    def show_spine_by_index(index: int) -> List[Tuple[Polyhedron_3, List[SpineMetric]]]:
        # keeping old views caused bugs when switching between spines
        # this sacrifices saving camera position but oh well
        spine_previews[index].create_views()
        display(spine_previews[index].widget)

        # selected spine meshes and metrics
        selection.clear()
        selection.extend([(selected_preview.spine_mesh, selected_preview.metrics)
                          for selected_preview in spine_previews if selected_preview.is_selected])
        return selection

    slider = widgets.IntSlider(min=0, max=len(spine_meshes) - 1)
    navigation_buttons = _make_navigation_widget(slider)

    return widgets.VBox([navigation_buttons,
                         widgets.interactive(show_spine_by_index,
                                             index=slider)])


def interactive_segmentation(mesh: Polyhedron_3, correspondence,
                             reverse_correspondence,
                             skeleton_graph) -> widgets.Widget:
    vertices, facets = _mesh_to_v_f(mesh)

    slider = widgets.FloatLogSlider(min=-3.0, max=0.0, step=0.01, value=-1.0,
                                    continuous_update=False)
    correction_slider = widgets.IntSlider(min=-6, max=6, value=0,
                                          continuous_update=False)
    plot = mp.plot(vertices, facets)

    def do_segmentation(sensitivity=0.15, correction=0):
        segmentation = segmentation_by_distance(mesh, correspondence,
                                                reverse_correspondence,
                                                skeleton_graph, 1 - sensitivity)
        segmentation = correct_segmentation(segmentation, mesh, correction)
        plot.update_object(colors=_segmentation_to_colors(vertices, segmentation))

        return segmentation

    return widgets.interactive(do_segmentation, sensitivity=slider,
                               correction=correction_slider)


def show_segmented_mesh(mesh: Polyhedron_3, segmentation: Segmentation):
    vertices, facets = _mesh_to_v_f(mesh)
    colors = _segmentation_to_colors(vertices, segmentation)
    mp.plot(vertices, facets, c=colors)


def show_sliced_image(image: np.ndarray, x: int, y: int, z: int,
                      mask: np.ndarray = None, mask_opacity=0.5,
                      cmap="gray", title=""):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10),
                           gridspec_kw={
                               'width_ratios': [image.shape[2],
                                                image.shape[1]],
                               'height_ratios': [image.shape[2],
                                                 image.shape[0]]
                           })

    if title != "":
        fig.suptitle(title)

    data_x = image[:, x, :]
    data_y = image[y, :, :].transpose()
    data_z = image[:, :, z]

    mask_x = None
    mask_y = None
    mask_z = None
    if mask is not None:
        mask_x = mask[:, x, :]
        mask_y = mask[y, :, :].transpose()
        mask_z = mask[:, :, z]

    ax[0, 0].axis("off")
    _show_image(ax[1, 0], data_x, mask=mask_x, mask_opacity=mask_opacity, title=f"X = {x}", cmap=cmap)
    _show_image(ax[0, 1], data_y, mask=mask_y, mask_opacity=mask_opacity, title=f"Y = {y}", cmap=cmap)
    _show_image(ax[1, 1], data_z, mask=mask_z, mask_opacity=mask_opacity, title=f"Z = {z}", cmap=cmap)

    _show_cross_planes(ax[1, 0], z, y, data_x.shape, "blue", "green", "red")
    _show_cross_planes(ax[0, 1], x, z, data_y.shape, "red", "blue", "green")
    _show_cross_planes(ax[1, 1], x, y, data_z.shape, "red", "green", "blue")

    plt.tight_layout()
    plt.show()


def show_3d_image(data: np.ndarray, cmap="gray"):
    @widgets.interact(
        x=widgets.IntSlider(min=0, max=data.shape[1] - 1, continuous_update=False),
        y=widgets.IntSlider(min=0, max=data.shape[0] - 1, continuous_update=False),
        z=widgets.IntSlider(min=0, max=data.shape[2] - 1, continuous_update=False),
        layout=widgets.Layout(width='500px'))
    def display_slice(x, y, z):
        show_sliced_image(data, x, y, z, cmap=cmap)

    return display_slice


class Image3DRenderer:
    image: np.ndarray
    title: str

    _x: int
    _y: int
    _z: int

    def __init__(self, image: np.ndarray = np.zeros(0), title: str = "Title"):
        self._x = -1
        self._y = -1
        self._z = -1
        self.image = image
        self.title = title

    def show(self, cmap="gray"):
        shape = self.image.shape

        if self._x < 0:
            self._x = shape[1] // 2
        if self._y < 0:
            self._y = shape[0] // 2
        if self._z < 0:
            self._z = shape[2] // 2

        @widgets.interact(x=widgets.IntSlider(value=self._x, min=0,
                                              max=shape[1] - 1,
                                              continuous_update=False),
                          y=widgets.IntSlider(value=self._y, min=0,
                                              max=shape[0] - 1,
                                              continuous_update=False),
                          z=widgets.IntSlider(value=self._z, min=0,
                                              max=shape[2] - 1,
                                              continuous_update=False))
        def display_slice(x, y, z):
            self._x = x
            self._y = y
            self._z = z
            self._display_slice(x, y, z, cmap)

        return display_slice

    def _display_slice(self, x, y, z, cmap):
        show_sliced_image(self.image, x, y, z, cmap=cmap, title=self.title)


class MaskedImage3DRenderer(Image3DRenderer):
    mask: np.ndarray
    _mask_opacity: float

    def __init__(self, image: np.ndarray = np.zeros(0),
                 mask: np.ndarray = np.zeros(0), title: str = "Title"):
        super().__init__(image, title)
        self.mask = mask
        self._mask_opacity = 1

    def _display_slice(self, x, y, z, cmap):
        @widgets.interact(mask_opacity=widgets.FloatSlider(min=0, max=1,
                                                           value=self._mask_opacity,
                                                           step=0.1,
                                                           continuous_update=False))
        def display_slice_with_mask(mask_opacity):
            self._mask_opacity = mask_opacity
            show_sliced_image(self.image, x, y, z,
                              mask=self.mask, mask_opacity=mask_opacity,
                              cmap=cmap, title=self.title)


def interactive_binarization(image: np.ndarray) -> widgets.Widget:
    base_threshold_slider = widgets.IntSlider(min=0, max=255, value=127,
                                              continuous_update=False)
    weight_slider = widgets.IntSlider(min=0, max=100, value=5,
                                      continuous_update=False)
    block_size_slider = widgets.IntSlider(min=1, max=31, value=3, step=2,
                                          continuous_update=False)

    image_renderer = MaskedImage3DRenderer(title="Binarization Result")

    def show_binarization(base_threshold: int, weight: int, block_size: int) -> np.ndarray:
        result = local_threshold_3d(image, base_threshold=base_threshold,
                                    weight=weight / 100, block_size=block_size)
        image_renderer.image = image
        image_renderer.mask = result
        image_renderer.show()

        return result

    return widgets.interactive(show_binarization,
                               base_threshold=base_threshold_slider,
                               weight=weight_slider,
                               block_size=block_size_slider)


def select_connected_component_widget(binary_image: np.ndarray) -> widgets.Widget:
    # find connected components
    labels, num_of_components = measurements.label(binary_image)

    # sort labels by size
    unique, counts = np.unique(labels, return_counts=True)
    unique = unique.tolist()
    counts = counts.tolist()
    unique.sort(key=lambda x: counts[x], reverse=True)
    counts.sort(reverse=True)

    # filter background and too small labels
    used_labels = []
    for i, count in enumerate(counts):
        if count >= 10 and unique[i] != 0:
            used_labels.append(unique[i])

    image_renderer = Image3DRenderer(title="Selected Connected Component")

    label_index_slider = widgets.IntSlider(min=0, max=len(used_labels) - 1,
                                           continuous_update=False)
    navigation_buttons = _make_navigation_widget(label_index_slider)
    display(navigation_buttons)

    def show_component(label_index: int) -> np.ndarray:
        lbl = used_labels[label_index]

        component = np.zeros_like(binary_image)
        component[labels == lbl] = 255

        preview = binary_image.copy()
        preview[preview > 0] = 64
        preview[labels == lbl] = 255

        image_renderer.image = preview
        image_renderer.show()

        return component

    return widgets.interactive(show_component, label_index=label_index_slider)


def grouping_in_3d_widget(grouping: SpineGrouping,
                          spine_dataset: SpineMeshDataset,
                          metrics_dataset: SpineMetricDataset,
                          distance_metric=None) -> widgets.Widget:
    spine_previews_by_cluster = {label: {} for label in grouping.group_labels}
    colors = grouping.colors

    def show_spine_by_group_label(label: str):
        def show_spine_by_name(spine_name: str):
            if spine_name not in spine_previews_by_cluster[label]:
                preview = SpinePreview(spine_dataset.spine_meshes[spine_name],
                                       spine_dataset.spine_v_f[spine_name],
                                       spine_dataset.get_dendrite_v_f(
                                           spine_name),
                                       metrics_dataset.row(spine_name),
                                       spine_name, colors[label][:3])
                spine_previews_by_cluster[label][spine_name] = preview

            # keeping old views caused bugs when switching between spines
            # this sacrifices saving camera position but oh well
            spine_previews_by_cluster[label][spine_name].create_views()
            display(spine_previews_by_cluster[label][spine_name].widget)
        spine_name_dropdown = widgets.Dropdown(options=grouping.get_sorted_group(label),
                                               description="Spine:")
        display(widgets.interactive(show_spine_by_name, spine_name=spine_name_dropdown))

    group_label_dropdown = widgets.Dropdown(options=grouping.sorted_group_labels,
                                            description="Group:")

    return widgets.interactive(show_spine_by_group_label, label=group_label_dropdown)


# def new_clusterization_widget(clusterizer: SpineClusterizer,
#                               spine_meshes: Dict[str, V_F]) -> widgets.Widget:
#     def show_grid_by_cluster_index(cluster_index: int):
#         # extract representative cluster meshes
#         cluster = [(clusterizer.get_spine_reduced_coord(spine_name),
#                     spine_meshes[spine_name])
#                    for spine_name in clusterizer.get_representative_samples(cluster_index, 9)]
#         # sort by y
#         cluster.sort(key=lambda x: x[0][1])
#         # separate into rows
#         grid_size = int(np.ceil(np.sqrt(len(cluster))))
#         rows = [cluster[i:i + grid_size] for i in range(0, len(cluster), grid_size)]
#
#         # make rendering grid
#         grid_widget = widgets.VBox(
#             [widgets.HBox([make_viewer(*spine_mesh, None, 100, 100)._renderer
#                            for (_, spine_mesh) in row])
#              for row in rows])
#
#         display(widgets.HBox([clusterizer.show({cluster_index}),
#                               grid_widget]))
#
#     cluster_slider = widgets.IntSlider(min=0, max=max(clusterizer.num_of_clusters - 1, 0))
#     cluster_navigation_buttons = _make_navigation_widget(cluster_slider)
#
#     return widgets.VBox([cluster_navigation_buttons,
#                          widgets.interactive(show_grid_by_cluster_index,
#                                              cluster_index=cluster_slider)])


def new_new_clusterization_widget(grouping: SpineGrouping,
                                  spine_dataset: SpineMeshDataset) -> widgets.Widget:
    colors = {}

    def show_dendrite_by_name(dendrite_name: str):
        if dendrite_name not in colors:
            colors[dendrite_name] = _grouping_to_colors(
                spine_dataset.dendrite_v_f[dendrite_name][0],
                spine_dataset.spine_meshes,
                grouping)

        viewer = make_viewer()
        viewer.add_mesh(*spine_dataset.dendrite_v_f[dendrite_name], colors[dendrite_name])
        # for spine_name in spine_dataset.dendrite_to_spines[dendrite_name]:
        #     viewer.add_mesh(*spine_dataset.spine_v_f[spine_name])

        display(viewer._renderer)

    dendrite_name_dropdown = widgets.Dropdown(
        options=list(spine_dataset.dendrite_meshes.keys()),
        description="Dendrite:"
    )

    return widgets.interactive(show_dendrite_by_name, dendrite_name=dendrite_name_dropdown)


def representative_spines_widget(grouping: SpineGrouping,
                                 spine_dataset: SpineMeshDataset,
                                 metrics_dataset: SpineMetricDataset,
                                 num_of_samples: int = 3,
                                 distance: Callable = euclidean) -> widgets.Widget:
    representatives = grouping.get_representative_samples(metrics_dataset, num_of_samples, distance)
    colors = grouping.colors

    rows = []
    for label in grouping.sorted_group_labels:
        row = [widgets.Label(f"{label}:")]
        for name in representatives[label]:
            viewer = make_viewer(100, 100)
            v_f = spine_dataset.spine_v_f[name]
            viewer.add_mesh(*v_f, make_colors(v_f, colors[label][:3]))
            row.append(viewer._renderer)
        rows.append(widgets.HBox(row))
    return widgets.HBox([grouping.show(metrics_dataset), widgets.VBox(rows)])


def clustering_experiment_widget(spine_metrics: SpineMetricDataset,
                                 every_spine_metrics: SpineMetricDataset,
                                 spine_dataset: SpineMeshDataset,
                                 clusterizer_type,
                                 param_slider_type, param_name,
                                 param_min_value, param_max_value, param_step,
                                 static_params: Dict,
                                 score_function: Callable[[SpineClusterizer], float],
                                 dim_reduction: str = "pca",
                                 show_method: str = "tsne",
                                 classification: SpineGrouping = None,
                                 save_folder: str = "output/clusterization") -> widgets.Widget:
    # calculate score graph
    reduced_dim = 2 if dim_reduction else -1

    scores = {reduced_dim: []}

    # scores = {-1: [], 2: []}
    # pca_dim = spine_metrics.row_as_array(spine_metrics.spine_names[0]).size // 2
    # while pca_dim >= 2:
    #     scores[pca_dim] = []
    #     pca_dim //= 2

    num_of_steps = int(np.ceil((param_max_value - param_min_value) / param_step))
    param_values = [np.clip(param_min_value + param_step * i,
                    param_min_value, param_max_value) for i in range(num_of_steps)]

    for (dim, dim_scores) in scores.items():
        for value in param_values:
            scored_clusterizer = clusterizer_type(**{param_name: value}, **static_params, dim=dim,
                                                  reduction=dim_reduction)
            scored_clusterizer.set_show_method(show_method)
            scored_clusterizer.fit(spine_metrics)
            dim_scores.append(score_function(scored_clusterizer))

    peak = np.nanargmax(scores[reduced_dim])

    def export_score_graph(_: widgets.Button):
        create_dir(save_folder)
        filename = f"{save_folder}/score_graph.csv"
        with open(filename, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow([param_name] + param_values)
            writer.writerow(["score"] + scores[reduced_dim])
        print(f"Saved score graph to '{filename}'.")

    export_score_button = widgets.Button(description="Export Score Graph")
    export_score_button.on_click(export_score_graph)
    display(export_score_button)

    # reg = LinearRegression().fit(np.reshape(param_values, (-1, 1)), scores[2])

    param_slider = param_slider_type(min=param_min_value,
                                     max=param_max_value,
                                     value=param_values[peak],
                                     step=param_step,
                                     continuous_update=False)

    def show_clusterization(param_value) -> None:
        clusterizer = clusterizer_type(**{param_name: param_value}, **static_params, dim=reduced_dim,
                                       reduction=dim_reduction)
        clusterizer.set_show_method(show_method)
        clusterizer.fit(spine_metrics)

        score_graph = widgets.Output()
        with score_graph:
            plt.axvline(x=param_value, color='g', linestyle='-')
            plt.axhline(y=0, color='r', linestyle='-')
            for (dim, dim_scores) in scores.items():
                if dim == -1:
                    plot_label = f"no {dim_reduction}"
                else:
                    plot_label = f"{dim}d with {dim_reduction} "
                plt.plot(param_values, dim_scores, label=plot_label)

            # plt.plot(param_values, reg.predict([[param] for param in param_values]))

            plt.title(clusterizer_type.__name__)
            plt.xlabel(param_name)
            plt.ylabel("Score")
            # plt.ylim([-1, 1])
            plt.legend()
            # plt.rcParams["figure.figsize"] = (10, 10)
            plt.show()

        # export clusterization button
        def export_clusterization(_: widgets.Button):
            create_dir(save_folder)
            save_path = f"{save_folder}/{param_name}={param_value}" \
                        f"_{clusterizer.reduction}={clusterizer.dim}" \
                        f"_{clusterizer.grouping.num_of_groups}_clusters"
            create_dir(save_path)
            save_path += "/"

            clusterization_save_path = save_path + "clusterization.json"
            clusterizer.grouping.save(clusterization_save_path)
            print(f"Saved clusterization to \"{clusterization_save_path}\".")

            reduced_save_path = save_path + f"reduced_{dim_reduction}.csv"
            clusterizer.grouping.save_reduced(spine_metrics, reduced_save_path, dim_reduction)
            print(f"Saved reduced coordinates to \"{reduced_save_path}\".")

            classification_save_path = save_path + "classification.json"
            classification.save(classification_save_path)
            print(f"Saved classification to \"{classification_save_path}\".")

            classification_save_reduced_path = save_path + f"classification_reduced_{dim_reduction}.csv"
            classification.save_reduced(spine_metrics, classification_save_reduced_path, dim_reduction)
            print(f"Saved classification reduced coordinates to \"{classification_save_reduced_path}\".")

            distribution_save_path = save_path + "metric_distributions.csv"
            clusterizer.grouping.save_metric_distribution(every_spine_metrics, distribution_save_path)
            print(f"Saved metric distributions to \"{distribution_save_path}\".")

            clust_over_class_save_path = save_path + "intersection_clust_over_class.csv"
            clusterizer.grouping.intersection_ratios(classification, False).save(clust_over_class_save_path)
            class_over_clust_save_path = save_path + "intersection_class_over_clust.csv"
            classification.intersection_ratios(clusterizer.grouping, False).save(class_over_clust_save_path)
            class_over_clust_norm_save_path = save_path + "intersection_class_over_clust_norm.csv"
            classification.intersection_ratios(clusterizer.grouping, True).save(class_over_clust_norm_save_path)
            print(f'Saved intersection ratios to "{clust_over_class_save_path}", '
                  f'"{class_over_clust_save_path}", "{class_over_clust_norm_save_path}".')

        export_button = widgets.Button(description="Export Clusterization")
        export_button.on_click(export_clusterization)

        # clusterization inspector
        inspector = inspect_grouping_widget(clusterizer.grouping, spine_dataset, spine_metrics,
                                            every_spine_metrics, classification)

        display(widgets.VBox([widgets.HBox([clusterizer.grouping.show(spine_metrics), score_graph]),
                              export_button, inspector]))

    clusterization_result = widgets.interactive(show_clusterization,
                                                param_value=param_slider)

    navigation_buttons = _make_navigation_widget(param_slider, param_step)

    return widgets.VBox([navigation_buttons, clusterization_result])


def k_means_clustering_experiment_widget(spine_metrics: SpineMetricDataset,
                                         every_spine_metrics: SpineMetricDataset,
                                         spine_dataset: SpineMeshDataset,
                                         score_function: Callable[[SpineClusterizer], float],
                                         min_num_of_clusters: int = 2,
                                         max_num_of_clusters: int = 20,
                                         metric="euclidean",
                                         dim_reduction: str = "pca",
                                         show_method: str = "tsne",
                                         classification: SpineGrouping = None,
                                         save_folder: str = "output/clustering") -> widgets.Widget:
    min_num_of_clusters = max(min_num_of_clusters, 2)
    max_num_of_clusters = min(max_num_of_clusters, spine_metrics.num_of_spines)
    create_dir(save_folder)
    return clustering_experiment_widget(spine_metrics, every_spine_metrics,
                                        spine_dataset,
                                        KMeansSpineClusterizer,
                                        widgets.IntSlider, "num_of_clusters",
                                        min_num_of_clusters, max_num_of_clusters,
                                        1, {"metric": metric}, score_function,
                                        dim_reduction, show_method, classification, f"{save_folder}/kmeans")


def dbscan_clustering_experiment_widget(spine_metrics: SpineMetricDataset,
                                        every_spine_metrics: SpineMetricDataset,
                                        spine_dataset: SpineMeshDataset,
                                        score_function: Callable[[SpineClusterizer], float],
                                        metric="euclidean",
                                        min_eps: float = 2,
                                        max_eps: float = 20,
                                        eps_step: float = 0.1,
                                        dim_reduction: str = "pca",
                                        show_method: str = "tsne",
                                        classification: SpineGrouping = None,
                                        save_folder: str = "output/clustering") -> widgets.Widget:
    create_dir(save_folder)
    return clustering_experiment_widget(spine_metrics, every_spine_metrics,
                                        spine_dataset,
                                        DBSCANSpineClusterizer,
                                        widgets.FloatSlider, "eps",
                                        min_eps, max_eps, eps_step,
                                        {"metric": metric}, score_function,
                                        dim_reduction, show_method, classification, f"{save_folder}/dbscan")


def grouping_metric_distribution_widget(grouping: SpineGrouping,
                                        metrics: SpineMetricDataset) -> widgets.Widget:
    metric_distributions = []
    for metric in metrics.row(list(metrics.spine_names)[0]):
        distribution_graph = widgets.Output()
        with distribution_graph:
            data = []
            colors = grouping.colors
            for label, cluster in grouping.groups.items():
                cluster_metrics = metrics.get_spines_subset(cluster)
                metric_column = list(cluster_metrics.column(metric.name).values())
                data.append(metric.get_distribution(metric_column))
                if issubclass(metric.__class__, HistogramSpineMetric):
                    value = metric.get_distribution(metric_column)
                    left_edges = [(int(label) - 1) + j / len(value) for j in range(len(value))]
                    width = left_edges[1] - left_edges[0]
                    plt.bar(left_edges, value, align='edge', width=width, color=colors[label])
            if issubclass(metric.__class__, FloatSpineMetric):
                plt.boxplot(data)
            plt.title(metric.name)
            plt.show()
        metric_distributions.append(distribution_graph)

    # slider = widgets.IntSlider(min=0, max=max(0, len(metric_distributions) - 1))
    # navigation_buttons = _make_navigation_widget(slider)
    # return widgets.VBox([navigation_buttons,
    #                      widgets.interactive(lambda index: display(metric_distributions[index]),
    #                                          index=slider)])

    return widgets.HBox(metric_distributions, layout=widgets.Layout(width='3000px'))


def color_to_hex(color: Tuple[float, float, float, float]) -> str:
    b = [int(c * 255) for c in color]
    c = (b[0] << 16) + (b[1] << 8) + b[2]
    return "#" + f"{c:06x}"


def manual_classification_widget(meshes: SpineMeshDataset,
                                 metrics: SpineMetricDataset,
                                 classes: Iterable[str],
                                 initial_classification: SpineGrouping = None) -> widgets.Widget:
    if initial_classification is None:
        initial_classification = SpineGrouping(meshes.spine_names,
                                               {class_name: set() for class_name in classes},
                                               "Unclassified")
    result_grouping = initial_classification
    colors = result_grouping.colors

    spine_names_list = list(meshes.spine_names)
    unclassified = result_grouping.outlier_group
    spine_names_list.sort(key=lambda x: x not in unclassified)

    spine_name = [""]

    preview = []

    def class_button_callback(clicked_button: widgets.Button) -> None:
        class_name = clicked_button.description

        # remove spine from current class
        current_class = result_grouping.get_group(spine_name[0])
        if current_class != result_grouping.outliers_label:
            result_grouping.groups[current_class].remove(spine_name[0])

        # add spine to new class
        result_grouping.groups[class_name].add(spine_name[0])

        # update preview spine color (if last spine, needs to be updated!)
        preview[0].spine_color = colors[class_name][:3]

        # move to next spine
        spine_index_slider.value += 1

    # create classification buttons
    class_buttons = []
    for class_name in classes:
        button = widgets.Button(description=class_name)
        button.style.button_color = color_to_hex(colors[class_name])
        button.style.text_color = "#FFFFFF"
        button.on_click(class_button_callback)
        class_buttons.append(button)
    class_buttons_box = widgets.HBox(class_buttons)

    def show_spine(spine_index: int) -> SpineGrouping:
        spine_name[0] = spine_names_list[spine_index]
        name = spine_name[0]
        preview.clear()
        preview.append(SpinePreview(meshes.spine_meshes[name], meshes.spine_v_f[name],
                                    meshes.get_dendrite_v_f(name), metrics.row(name),
                                    name, result_grouping.get_color(name)[:3]))
        display(preview[0].widget)
        return result_grouping

    spine_index_slider = widgets.IntSlider(max=max(0, len(spine_names_list) - 1))
    spine_classification = widgets.interactive(show_spine, spine_index=spine_index_slider)

    navigation_buttons = _make_navigation_widget(spine_index_slider)

    return widgets.VBox([widgets.VBox([class_buttons_box, navigation_buttons]),
                         spine_classification])


def intersection_ratios_mean_distance(a: SpineGrouping, b: SpineGrouping, normalize: bool = True) -> float:
    intersections = a.intersection_ratios(b, normalize)

    a_labels = list(intersections.keys())
    b_labels = list(b.group_labels_with_outliers)

    mean_distance = 0
    num = 0
    for i in range(len(a_labels) - 1):
        for j in range(i + 1, len(a_labels)):
            row_i = np.array([intersections[a_labels[i]][b_label] for b_label in b_labels])
            row_j = np.array([intersections[a_labels[j]][b_label] for b_label in b_labels])
            mean_distance += np.linalg.norm(row_i - row_j)
            num += 1
    mean_distance /= num

    return mean_distance


def grouping_intersection_widget(a: SpineGrouping, b: SpineGrouping, normalize: bool = True) -> widgets.Widget:
    intersections = a.intersection_ratios(b, normalize)

    # calculate mean distance
    mean_distance = intersection_ratios_mean_distance(a, b, normalize)
    mean_distance_label = widgets.Label(f"Mean distance b/w distributions: {mean_distance:.2f}")

    # generate pie charts
    b_colors = b.colors_with_outliers

    pie_charts = {}
    for a_label in intersections.keys():
        pie_chart_widget = widgets.Output()
        with pie_chart_widget:
            a_intersection = intersections[a_label].copy()
            plt.bar(range(len(a_intersection)), list(a_intersection.values()),
                    tick_label=list(a_intersection.keys()),
                    color=[b_colors[label] for label in a_intersection.keys()])
            plt.show()
            # remove zero-length segments
            for key, value in list(a_intersection.items()):
                if value == 0:
                    del a_intersection[key]
            plt.pie(list(a_intersection.values()),
                    labels=list(a_intersection.keys()), autopct='%1.1f%%', pctdistance=0.85,
                    colors=[b_colors[label] for label in a_intersection.keys()],
                    normalize=True)
            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)
            plt.show()
        pie_charts[a_label] = pie_chart_widget

    all_charts_widget = widgets.HBox([widgets.VBox([widgets.Label(f"{a_label}:"), pie_chart_widget])
                                      for a_label, pie_chart_widget in pie_charts.items()])

    return widgets.VBox([mean_distance_label, all_charts_widget])

    # # generate table
    # for class_label in class_labels:
    #     class_label_widget = widgets.Label(str(class_label))
    #     class_label_widget.st = b_colors[class_label][:3]
    #     grid_items.append(class_label_widget)
    # for a_label in clustering.group_labels:
    #     grid_items.append(widgets.Label(str(a_label)))
    #     for class_label in class_labels:
    #         grid_items.append(widgets.Label(f"{intersections[a_label][class_label]:.2f}"))
    #
    # return widgets.GridBox(grid_items, layout=widgets.Layout(grid_template_columns=f"repeat({classification.num_of_groups + 1}, 100px)"))


def consensus_widget(groupings: List[SpineGrouping]) -> widgets.Widget:
    def compare(a, b) -> int:
        label_a = merged_grouping.get_group(a)
        votes_a = sum(1 for grouping in groupings
                      if grouping.get_group(a) == label_a)
        label_b = merged_grouping.get_group(b)
        votes_b = sum(1 for grouping in groupings
                      if grouping.get_group(b) == label_b)
        if votes_a < votes_b:
            return -1
        if votes_a > votes_b:
            return 1

        # sort by size if equal votes
        samples_a = merged_grouping.get_group_size(label_a)
        samples_b = merged_grouping.get_group_size(label_b)
        return np.sign(samples_a - samples_b)

    for grouping in groupings:
        grouping.outliers_label = "Unclassified"

    merged_grouping = SpineGrouping.merge(groupings, outliers_label="Unclassified")
    labels = list(merged_grouping.group_labels)
    labels.sort(key=lambda label: len(merged_grouping.groups[label]), reverse=True)

    sorted_spines = list(merged_grouping.samples)
    sorted_spines.sort(key=cmp_to_key(compare), reverse=True)

    colors = merged_grouping.colors_with_outliers

    # legend = []
    # for label in merged_grouping.groups:
    #     button = widgets.Button(description=label)
    #     button.style.button_color = color_to_hex(colors[label])
    #     button.style.text_color = "#FFFFFF"
    #     legend.append(button)
    # class_buttons_box = widgets.HBox(class_buttons)

    grid_items = [widgets.Widget()]
    for i, _ in enumerate(groupings):
        grid_items.append(widgets.Label(str(i + 1)))

    for i, spine_name in enumerate(sorted_spines):
        spine_name_label = widgets.Label(str(i + 1))
        spine_name_label.layout.width = "30px"
        grid_items.append(spine_name_label)
        for grouping in groupings:
            group_label = grouping.get_group(spine_name)
            color = color_to_hex(colors[group_label])
            rect = widgets.HTML(value=f'<svg width="30" height="15"><rect width="30" height="15" style="fill:{color};stroke:black;stroke-width:2"/></svg>')
            grid_items.append(rect)
            # button = widgets.Button(disabled=True)
            # button.style.button_color = color_to_hex(colors[group_label])
            # button.layout.width = "10px"
            # row.append(button)

    return widgets.GridBox(grid_items, layout=widgets.Layout(grid_template_columns=f"repeat({len(groupings) + 1}, 30px)"))


def dendrite_segmentation_view_widget(spine_dataset: SpineMeshDataset) -> widgets.Widget:
    def show_dendrite_by_name(dendrite_name: str):
        dendrite_v_f = spine_dataset.dendrite_v_f[dendrite_name]
        dendrite_colors = np.ndarray((len(dendrite_v_f[0]), 3))
        dendrite_colors[:] = \
            _segmentation_to_colors(dendrite_v_f[0],
                                    spines_to_segmentation([spine_dataset.spine_meshes[spine] for spine in spine_dataset.dendrite_to_spines[dendrite_name]]))

        mesh_viewer = make_viewer(600, 600)
        mesh_viewer.add_mesh(*dendrite_v_f, dendrite_colors)

        display(widgets.HBox([mesh_viewer._renderer]))

    names = list(spine_dataset.dendrite_names)
    names.sort()
    dendrite_names_dropdown = widgets.Dropdown(options=names, description="Dendrite:")

    return widgets.interactive(show_dendrite_by_name, dendrite_name=dendrite_names_dropdown)


def spine_dataset_view_widget(spine_dataset: SpineMeshDataset,
                              metrics_dataset: SpineMetricDataset,
                              spine_color: Color = RED) -> widgets.Widget:
    def show_spine_by_name(spine_name: str):
        spine_mesh = spine_dataset.spine_meshes[spine_name]
        spine_v_f = spine_dataset.spine_v_f[spine_name]
        dendrite_v_f = spine_dataset.get_dendrite_v_f(spine_name)
        metrics = metrics_dataset.row(spine_name)
        spine_preview = SpinePreview(spine_mesh, spine_v_f, dendrite_v_f, metrics,
                                     spine_name, spine_color)
        display(spine_preview.widget)
    names = list(spine_dataset.spine_names)
    names.sort()
    spine_names_dropdown = widgets.Dropdown(options=names,
                                            description="Spine:")
    return widgets.interactive(show_spine_by_name, spine_name=spine_names_dropdown)


def spine_chords_widget(spine_dataset: SpineMeshDataset, scaled_spine_dataset: SpineMeshDataset,
                        dataset_path: str, num_of_chords: int = 3000,
                        num_of_bins: int = 100) -> widgets.Widget:
    chord_metrics = {}
    scaled_chord_metrics = {}

    def show_spine_by_name(spine_name: str):
        if spine_name in chord_metrics:
            chord_metric = chord_metrics[spine_name]
            scaled_chord_metric = scaled_chord_metrics[spine_name]
        else:
            chord_metric = OldChordDistributionSpineMetric(spine_dataset.spine_meshes[spine_name],
                                                           num_of_chords=num_of_chords, num_of_bins=num_of_bins)
            chord_metrics[spine_name] = [chord_metric]
            scaled_chord_metric = OldChordDistributionSpineMetric(scaled_spine_dataset.spine_meshes[spine_name],
                                                                  num_of_chords=num_of_chords, num_of_bins=num_of_bins)
            scaled_chord_metrics[spine_name] = [chord_metric]
        view = mp.Viewer({})
        _add_line_set_to_viewer(view, scaled_chord_metric.chords)
        _add_mesh_to_viewer_as_wireframe(view, scaled_spine_dataset.spine_v_f[spine_name])

        display(widgets.HBox([view._renderer, chord_metric.show()]))

    def export_callback(_: widgets.Button) -> None:
        save_path = f"{dataset_path}/chords_{num_of_chords}_chords_{num_of_bins}_bins.csv"
        SpineMetricDataset(chord_metrics).save_as_array(save_path)
        print(f"Saved histograms to \"{save_path}\"")

    export_button = widgets.Button(description="Export Histograms")
    export_button.on_click(export_callback)

    names = list(spine_dataset.spine_names)
    names.sort()
    spine_names_dropdown = widgets.Dropdown(options=names,
                                            description="Spine:")
    return widgets.VBox([widgets.interactive(show_spine_by_name, spine_name=spine_names_dropdown), export_button])


def view_skeleton_widget(scaled_spine_dataset: SpineMeshDataset) -> widgets.Widget:
    def show_dendrite_by_name(dendrite_name: str):
        dendrite_mesh = scaled_spine_dataset.dendrite_meshes[dendrite_name]
        dendrite_v_f = scaled_spine_dataset.dendrite_v_f[dendrite_name]

        # get skeleton
        skeleton_polylines = Polylines()
        correspondence_polylines = Polylines()
        surface_mesh_skeletonization(dendrite_mesh, skeleton_polylines,
                                     correspondence_polylines)

        # make viewers
        w = 600
        h = 600

        mesh_viewer = make_viewer(w, h)
        mesh_viewer.add_mesh(*scaled_spine_dataset.dendrite_v_f[dendrite_name])

        skeleton_viewer = make_viewer(w, h)
        skeleton_line_set = polylines_to_line_set(skeleton_polylines)
        _add_line_set_to_viewer(skeleton_viewer, skeleton_line_set)

        skeleton_mesh_viewer = make_viewer(w, h)
        _add_line_set_to_viewer(skeleton_mesh_viewer, polylines_to_line_set(skeleton_polylines))
        _add_mesh_to_viewer_as_wireframe(skeleton_mesh_viewer, dendrite_v_f)

        display(widgets.HBox([mesh_viewer._renderer, skeleton_viewer._renderer,
                              skeleton_mesh_viewer._renderer]))

    names = list(scaled_spine_dataset.dendrite_names)
    names.sort()
    dendrite_names_dropdown = widgets.Dropdown(options=names, description="Dendrite:")
    
    return widgets.interactive(show_dendrite_by_name, dendrite_name=dendrite_names_dropdown)


def inspect_grouping_widget(grouping: SpineGrouping, spine_dataset: SpineMeshDataset,
                            metrics_dataset: SpineMetricDataset, all_metrics_dataset: SpineMetricDataset,
                            reference_grouping: SpineGrouping) -> widgets.Widget:
    inspectors = {}
    inspector_names = ["Metric Distribution", "Reference Grouping Intersection",
                       "View Spines in 3D", "Representative Spines"]

    def generate_inspector(inspector_index) -> widgets.Widget:
        if inspector_index == 0:
            return grouping_metric_distribution_widget(grouping, all_metrics_dataset)
        elif inspector_index == 1:
            if reference_grouping is not None:
                intersection_widgets = [
                    reference_grouping.show(metrics_dataset),
                    widgets.Label("-- Clusters Over Classes --"),
                    grouping_intersection_widget(grouping, reference_grouping, False),
                    widgets.Label("-- Classes Over Clusters, Non-normalized --"),
                    grouping_intersection_widget(reference_grouping, grouping, False),
                    widgets.Label("-- Clusters Over Classes, Normalized --"),
                    grouping_intersection_widget(reference_grouping, grouping, True)
                ]
            else:
                intersection_widgets = [widgets.Label("No reference grouping provided!")]
            return widgets.VBox(intersection_widgets)
        elif inspector_index == 2:
            return grouping_in_3d_widget(grouping, spine_dataset, metrics_dataset)
        elif inspector_index == 3:
            return representative_spines_widget(grouping, spine_dataset, metrics_dataset)

    def show_inspector(inspector_index):
        if inspector_index not in inspectors:
            inspectors[inspector_index] = generate_inspector(inspector_index)
        display(inspectors[inspector_index])

    inspector_dropdown = widgets.Dropdown(
        options=[(name, i) for i, name in enumerate(inspector_names)])

    return widgets.interactive(show_inspector, inspector_index=inspector_dropdown)


def inspect_saved_groupings_widget(folder_path: str, spine_dataset: SpineMeshDataset,
                                   all_metrics_dataset: SpineMetricDataset,
                                   chord_metric_dataset: SpineMetricDataset,
                                   classic_metrics_dataset: SpineMetricDataset,
                                   reference_grouping: SpineGrouping,
                                   grouping_file_pattern: str = "**/*.json") -> widgets.Widget:
    path = Path(folder_path)

    grouping_paths = [str(grouping_path) for grouping_path in path.glob(grouping_file_pattern)]
    grouping_paths.sort()
    groupings = {grouping_path: SpineGrouping().load(grouping_path)
                 for grouping_path in grouping_paths}

    metrics = [all_metrics_dataset, chord_metric_dataset, classic_metrics_dataset]
    metric_names = ["Combined", "Chord Length Histogram", "Classic Metrics"]

    def inspect_grouping(grouping_path: str):
        if grouping_path is None:
            print("No groupings found!")
            return
        grouping = groupings[grouping_path]
        header = widgets.HBox([widgets.VBox([widgets.Label(f"{name}:"),
                                             grouping.show(dataset)])
                               for name, dataset in zip(metric_names, metrics)])
        display(header)

        def inspect_grouping_by_metric(metrics_set: int):
            display(inspect_grouping_widget(grouping, spine_dataset,
                                            metrics[metrics_set], all_metrics_dataset,
                                            reference_grouping))
            
        metrics_dropdown = widgets.Dropdown(options=[(name, i) for i, name in enumerate(metric_names)])
        display(widgets.interactive(inspect_grouping_by_metric, metrics_set=metrics_dropdown))

    groupings_dropdown = widgets.Dropdown(options=grouping_paths)

    return widgets.interactive(inspect_grouping, grouping_path=groupings_dropdown)

    