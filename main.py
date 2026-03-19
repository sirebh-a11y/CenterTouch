import sys
import math
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QTextEdit, QFileDialog, QTableWidget,
    QTableWidgetItem, QGroupBox, QComboBox, QDoubleSpinBox,
    QMessageBox, QCheckBox, QTabWidget, QSizePolicy, QScrollArea
)


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class CircleFitResult:
    center_3d: np.ndarray
    radius: float
    rms: float
    max_residual: float
    local_plane_normal: np.ndarray
    local_plane_point: np.ndarray
    num_points: int


@dataclass
class PlaneFitResult:
    normal: np.ndarray
    point: np.ndarray
    rms: float
    max_residual: float
    area_indicator: float
    num_points: int


@dataclass
class HoleInputResult:
    center_raw: np.ndarray
    source: str  # "points" or "center"
    circle_fit: Optional[CircleFitResult]


@dataclass
class FrameResult:
    origin: np.ndarray
    R: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray


@dataclass
class QualityReport:
    status: str
    lines: List[str]


# ============================================================
# MATH UTILITIES
# ============================================================

EPS = 1e-10


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < EPS:
        raise ValueError("Vettore nullo o quasi nullo.")
    return v / n


def orient_real_plane_normal(normal: np.ndarray, force_flip: bool = False) -> np.ndarray:
    n = normalize(normal.copy())
    if force_flip:
        n = -n
    else:
        if n[2] < 0:
            n = -n
    return n


def ensure_right_handed(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = np.column_stack((X, Y, Z))
    if np.linalg.det(R) < 0:
        Y = -Y
    return X, Y, Z


def fit_plane(points: np.ndarray) -> PlaneFitResult:
    if len(points) < 3:
        raise ValueError("Servono almeno 3 punti per il piano.")

    pts = np.asarray(points, dtype=float)
    centroid = pts.mean(axis=0)

    A = pts - centroid
    _, s, vh = np.linalg.svd(A, full_matrices=False)
    normal = normalize(vh[-1])

    distances = (pts - centroid) @ normal
    rms = float(np.sqrt(np.mean(distances ** 2)))
    max_residual = float(np.max(np.abs(distances)))

    # Indicatore grezzo di area coperta dai punti sul piano:
    # usa i due principali assi del piano e area box 2D
    u = normalize(vh[0])
    v = normalize(vh[1])
    coords_u = (pts - centroid) @ u
    coords_v = (pts - centroid) @ v
    area_indicator = float((coords_u.max() - coords_u.min()) * (coords_v.max() - coords_v.min()))

    return PlaneFitResult(
        normal=normal,
        point=centroid,
        rms=rms,
        max_residual=max_residual,
        area_indicator=area_indicator,
        num_points=len(points),
    )


def project_point_on_plane(point: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    n = normalize(plane_normal)
    d = np.dot(point - plane_point, n)
    return point - d * n


def project_points_to_plane_basis(points: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray):
    """
    Restituisce coordinate 2D dei punti su una base ortonormale del piano.
    """
    n = normalize(plane_normal)
    # sceglie un vettore non parallelo a n
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, n)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    u = normalize(np.cross(n, ref))
    v = normalize(np.cross(n, u))

    proj_pts = np.array([project_point_on_plane(p, plane_point, n) for p in points])
    rel = proj_pts - plane_point
    pts_2d = np.column_stack((rel @ u, rel @ v))

    return proj_pts, pts_2d, u, v


def fit_circle_2d(points_2d: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Fit algebrico cerchio 2D: (x-a)^2 + (y-b)^2 = r^2
    """
    if len(points_2d) < 3:
        raise ValueError("Servono almeno 3 punti per il fit del cerchio.")

    x = points_2d[:, 0]
    y = points_2d[:, 1]

    A = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
    b = x**2 + y**2

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b0, c = sol
    radius_sq = a*a + b0*b0 + c
    if radius_sq <= 0:
        raise ValueError("Fit del cerchio non valido: raggio^2 <= 0.")

    center = np.array([a, b0])
    radius = float(np.sqrt(radius_sq))
    residuals = np.sqrt((x - a)**2 + (y - b0)**2) - radius
    return center, radius, residuals


def fit_circle_3d(points: np.ndarray) -> CircleFitResult:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        raise ValueError("Servono almeno 3 punti per il foro.")

    plane = fit_plane(pts)
    _, pts_2d, u, v = project_points_to_plane_basis(pts, plane.point, plane.normal)
    center_2d, radius, residuals = fit_circle_2d(pts_2d)

    center_3d = plane.point + center_2d[0] * u + center_2d[1] * v

    return CircleFitResult(
        center_3d=center_3d,
        radius=radius,
        rms=float(np.sqrt(np.mean(residuals ** 2))),
        max_residual=float(np.max(np.abs(residuals))),
        local_plane_normal=plane.normal,
        local_plane_point=plane.point,
        num_points=len(points),
    )


def build_frame_from_holes_and_plane(F1: np.ndarray, F2: np.ndarray, plane_normal: np.ndarray) -> FrameResult:
    origin = F1
    Z = normalize(plane_normal)
    X = normalize(F2 - F1)

    cross_mag = np.linalg.norm(np.cross(Z, X))
    if cross_mag < 1e-6:
        raise ValueError("Asse X quasi parallelo a Z: frame non stabile.")

    Y = normalize(np.cross(Z, X))
    X = normalize(np.cross(Y, Z))

    X, Y, Z = ensure_right_handed(X, Y, Z)
    R = np.column_stack((X, Y, Z))

    return FrameResult(origin=origin, R=R, X=X, Y=Y, Z=Z)


def rotation_matrix_to_euler_zyx_deg(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Restituisce angoli ZYX in gradi: (Rz, Ry, Rx)
    """
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)

    singular = sy < 1e-8
    if not singular:
        rz = math.atan2(R[1, 0], R[0, 0])
        ry = math.atan2(-R[2, 0], sy)
        rx = math.atan2(R[2, 1], R[2, 2])
    else:
        rz = math.atan2(-R[0, 1], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rx = 0.0

    return math.degrees(rz), math.degrees(ry), math.degrees(rx)


def homogeneous_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def format_vec(v: np.ndarray, prec: int = 6) -> str:
    return f"({v[0]:.{prec}f}, {v[1]:.{prec}f}, {v[2]:.{prec}f})"


def format_matrix(M: np.ndarray, prec: int = 6) -> str:
    rows = []
    for r in M:
        rows.append("[" + ", ".join(f"{x:.{prec}f}" for x in r) + "]")
    return "\n".join(rows)


# ============================================================
# QUALITY
# ============================================================

def build_quality_report(
    hole1: HoleInputResult,
    hole2: HoleInputResult,
    plane: PlaneFitResult,
    F1r: np.ndarray,
    F2r: np.ndarray,
    F1n_proj: np.ndarray,
    F2n_proj: np.ndarray,
    Zr: np.ndarray,
    Xr: np.ndarray,
    thresholds: dict
) -> QualityReport:
    lines = []
    severity = 0  # 0 OK, 1 WARNING, 2 CRITICAL

    # Piano
    lines.append(f"Piano reale: RMS={plane.rms:.6f}, Max={plane.max_residual:.6f}, AreaIndic={plane.area_indicator:.6f}")
    if plane.rms > thresholds["plane_rms_critical"]:
        lines.append("CRITICAL: errore RMS piano oltre soglia critica.")
        severity = max(severity, 2)
    elif plane.rms > thresholds["plane_rms_warning"]:
        lines.append("WARNING: errore RMS piano oltre soglia warning.")
        severity = max(severity, 1)

    if plane.area_indicator < thresholds["plane_area_warning"]:
        lines.append("WARNING: punti piano poco distribuiti, normale potenzialmente instabile.")
        severity = max(severity, 1)

    # Fori
    for idx, hole in [(1, hole1), (2, hole2)]:
        if hole.source == "points" and hole.circle_fit is not None:
            cf = hole.circle_fit
            lines.append(
                f"Foro {idx}: fit cerchio RMS={cf.rms:.6f}, Max={cf.max_residual:.6f}, "
                f"Raggio={cf.radius:.6f}, N={cf.num_points}"
            )
            if cf.rms > thresholds["hole_rms_critical"]:
                lines.append(f"CRITICAL: fit foro {idx} oltre soglia critica.")
                severity = max(severity, 2)
            elif cf.rms > thresholds["hole_rms_warning"]:
                lines.append(f"WARNING: fit foro {idx} oltre soglia warning.")
                severity = max(severity, 1)
        else:
            lines.append(f"Foro {idx}: centro inserito direttamente, nessun fit disponibile.")
            lines.append(f"WARNING: affidabilità del foro {idx} dipende dal dato esterno.")
            severity = max(severity, 1)

    # Distanza fori reale
    d_real = float(np.linalg.norm(F2r - F1r))
    d_nom = float(np.linalg.norm(F2n_proj - F1n_proj))
    diff_d = abs(d_real - d_nom)
    lines.append(f"Interasse nominale={d_nom:.6f}, reale={d_real:.6f}, delta={diff_d:.6f}")

    if d_real < thresholds["hole_distance_critical"]:
        lines.append("CRITICAL: distanza tra i fori troppo piccola, frame instabile.")
        severity = max(severity, 2)

    if diff_d > thresholds["distance_delta_critical"]:
        lines.append("CRITICAL: differenza interasse nominale/reale oltre soglia critica.")
        severity = max(severity, 2)
    elif diff_d > thresholds["distance_delta_warning"]:
        lines.append("WARNING: differenza interasse nominale/reale oltre soglia warning.")
        severity = max(severity, 1)

    # Quasi parallelismo X/Z
    cross_mag = float(np.linalg.norm(np.cross(Zr, Xr)))
    lines.append(f"Stabilità X vs Z: |Z x X|={cross_mag:.6f}")
    if cross_mag < thresholds["xz_cross_critical"]:
        lines.append("CRITICAL: asse X quasi parallelo a Z.")
        severity = max(severity, 2)
    elif cross_mag < thresholds["xz_cross_warning"]:
        lines.append("WARNING: asse X vicino al parallelismo con Z.")
        severity = max(severity, 1)

    status = "OK" if severity == 0 else ("WARNING" if severity == 1 else "CRITICAL")
    return QualityReport(status=status, lines=lines)


# ============================================================
# GUI HELPERS
# ============================================================

def set_table_headers(table: QTableWidget):
    table.setColumnCount(3)
    table.setHorizontalHeaderLabels(["X", "Y", "Z"])
    table.horizontalHeader().setStretchLastSection(True)


def set_cell_color(item: QTableWidgetItem, valid: bool):
    if valid:
        item.setBackground(QColor("white"))
    else:
        item.setBackground(QColor(255, 100, 100))


def parse_float_input(value: str) -> float:
    return float(value.strip().replace(",", "."))


def validate_cell(item: Optional[QTableWidgetItem]):
    if item is None:
        return

    txt = item.text().strip() if item.text() is not None else ""
    if txt == "":
        set_cell_color(item, True)
        return

    try:
        parse_float_input(txt)
        set_cell_color(item, True)
    except Exception:
        set_cell_color(item, False)


def read_points_from_table(table: QTableWidget) -> np.ndarray:
    pts = []
    flag_error = False
    for r in range(table.rowCount()):
        items = []
        texts = []
        for c in range(3):
            item = table.item(r, c)
            if item is None:
                item = QTableWidgetItem("")
                table.setItem(r, c, item)
            items.append(item)
            texts.append(item.text().strip() if item.text() is not None else "")

        if all(txt == "" for txt in texts):
            for item in items:
                set_cell_color(item, True)
            continue

        row_values = []
        for item, txt in zip(items, texts):
            if txt == "":
                set_cell_color(item, False)
                flag_error = True
                row_values.append(None)
                continue

            try:
                value = parse_float_input(txt)
                set_cell_color(item, True)
                row_values.append(value)
            except Exception:
                set_cell_color(item, False)
                flag_error = True
                row_values.append(None)
        if any(value is None for value in row_values):
            continue
        pts.append(row_values)
    if flag_error:
        raise ValueError("Errore: celle non valide evidenziate in rosso")
    return np.asarray(pts, dtype=float)


def create_empty_table(rows: int = 5) -> QTableWidget:
    table = QTableWidget(rows, 3)
    set_table_headers(table)
    table.itemChanged.connect(validate_cell)
    return table


class XYZInputRow(QWidget):
    def __init__(self, title: str):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel(title))

        self.x = QDoubleSpinBox()
        self.y = QDoubleSpinBox()
        self.z = QDoubleSpinBox()
        for w in (self.x, self.y, self.z):
            w.setRange(-1_000_000, 1_000_000)
            w.setDecimals(6)
            w.setSingleStep(0.1)
            w.setMinimumWidth(120)

        layout.addWidget(QLabel("X"))
        layout.addWidget(self.x)
        layout.addWidget(QLabel("Y"))
        layout.addWidget(self.y)
        layout.addWidget(QLabel("Z"))
        layout.addWidget(self.z)
        layout.addStretch(1)

    def value(self) -> np.ndarray:
        return np.array([self.x.value(), self.y.value(), self.z.value()], dtype=float)


class HoleInputWidget(QGroupBox):
    def __init__(self, title: str):
        super().__init__(title)
        layout = QVBoxLayout(self)

        self.mode = QComboBox()
        self.mode.addItems([
            "Punti tastati foro",
            "Centro foro già calcolato"
        ])
        layout.addWidget(QLabel("Modalità input"))
        layout.addWidget(self.mode)

        self.instructions = QLabel(
            "Metodo foro:\n"
            "- Se usi punti: inserisci 4-5 punti o più, ben distribuiti sul foro.\n"
            "- Se usi centro: inserisci il centro già calcolato.\n"
            "- La quota Z del foro NON viene usata direttamente: il centro verrà sempre proiettato sul piano reale."
        )
        self.instructions.setWordWrap(True)
        self.instructions.setStyleSheet("color: #333;")
        layout.addWidget(self.instructions)

        self.points_table = create_empty_table(6)
        layout.addWidget(self.points_table)

        btn_row = QHBoxLayout()
        self.add_row_btn = QPushButton("Aggiungi riga punti")
        self.remove_row_btn = QPushButton("Rimuovi riga punti")
        btn_row.addWidget(self.add_row_btn)
        btn_row.addWidget(self.remove_row_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.center_row = XYZInputRow("Centro foro")
        layout.addWidget(self.center_row)

        self.add_row_btn.clicked.connect(self.add_row)
        self.remove_row_btn.clicked.connect(self.remove_row)
        self.mode.currentIndexChanged.connect(self.update_visibility)
        self.update_visibility()

    def add_row(self):
        self.points_table.insertRow(self.points_table.rowCount())

    def remove_row(self):
        if self.points_table.rowCount() > 1:
            self.points_table.removeRow(self.points_table.rowCount() - 1)

    def update_visibility(self):
        is_points = self.mode.currentIndex() == 0
        self.points_table.setVisible(is_points)
        self.add_row_btn.setVisible(is_points)
        self.remove_row_btn.setVisible(is_points)
        self.center_row.setVisible(not is_points)

    def get_result(self) -> HoleInputResult:
        if self.mode.currentIndex() == 0:
            pts = read_points_from_table(self.points_table)
            if len(pts) < 3:
                raise ValueError(f"{self.title()}: servono almeno 3 punti.")
            circle = fit_circle_3d(pts)
            return HoleInputResult(
                center_raw=circle.center_3d,
                source="points",
                circle_fit=circle
            )
        else:
            center = self.center_row.value()
            return HoleInputResult(
                center_raw=center,
                source="center",
                circle_fit=None
            )


class PlaneInputWidget(QGroupBox):
    def __init__(self, title: str):
        super().__init__(title)
        layout = QVBoxLayout(self)

        info = QLabel(
            "Metodo piano:\n"
            "- Inserire 3, 4 o 5 punti del piano reale.\n"
            "- I punti devono essere il più possibile distribuiti.\n"
            "- La normale del piano definisce l'asse Z reale."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #333;")
        layout.addWidget(info)

        self.table = create_empty_table(5)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.add_row_btn = QPushButton("Aggiungi riga piano")
        self.remove_row_btn = QPushButton("Rimuovi riga piano")
        btn_row.addWidget(self.add_row_btn)
        btn_row.addWidget(self.remove_row_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.add_row_btn.clicked.connect(self.add_row)
        self.remove_row_btn.clicked.connect(self.remove_row)

    def add_row(self):
        self.table.insertRow(self.table.rowCount())

    def remove_row(self):
        if self.table.rowCount() > 1:
            self.table.removeRow(self.table.rowCount() - 1)

    def get_result(self) -> PlaneFitResult:
        pts = read_points_from_table(self.table)
        return fit_plane(pts)


class ThresholdsWidget(QGroupBox):
    def __init__(self):
        super().__init__("Soglie qualità")
        layout = QGridLayout(self)

        self.widgets = {}
        rows = [
            ("plane_rms_warning", "RMS piano warning", 0.05),
            ("plane_rms_critical", "RMS piano critical", 0.15),
            ("plane_area_warning", "Area indicativa piano warning", 1.0),
            ("hole_rms_warning", "RMS foro warning", 0.05),
            ("hole_rms_critical", "RMS foro critical", 0.15),
            ("hole_distance_critical", "Interasse minimo critical", 1.0),
            ("distance_delta_warning", "Delta interasse warning", 0.20),
            ("distance_delta_critical", "Delta interasse critical", 0.50),
            ("xz_cross_warning", "|ZxX| warning", 0.10),
            ("xz_cross_critical", "|ZxX| critical", 0.01),
        ]

        for r, (key, label, default) in enumerate(rows):
            sb = QDoubleSpinBox()
            sb.setRange(0.0, 1_000_000.0)
            sb.setDecimals(6)
            sb.setValue(default)
            sb.setSingleStep(0.01)
            self.widgets[key] = sb
            layout.addWidget(QLabel(label), r, 0)
            layout.addWidget(sb, r, 1)

    def values(self) -> dict:
        return {k: w.value() for k, w in self.widgets.items()}


# ============================================================
# MAIN WINDOW
# ============================================================

class MeltioFrameTool(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meltio Space - Tool di centraggio CAD → reale")
        self.resize(1300, 900)

        main_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        main_layout.addWidget(scroll_area)

        content = QWidget()
        scroll_area.setWidget(content)
        content_layout = QVBoxLayout(content)

        title = QLabel("Tool -BASIC- per centraggio CAD → reale")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        content_layout.addWidget(title)

        instructions = QLabel(
            "Riferimenti metrologici usati dal software:\n"
            "- Origine reale: centro foro 1 proiettato sul piano reale\n"
            "- Asse X reale: direzione foro 1 → foro 2 dopo proiezione\n"
            "- Asse Z reale: normale del piano reale tastato\n"
            "- Asse Y reale: calcolato automaticamente (sistema destrorso)\n"
            "- La Z dei fori tastati non viene mai usata direttamente\n"
            "- Lato CAD: i centri foro nominali vengono proiettati sul piano nominale Z=h, assunto parallelo a XY\n"
            "- Output principale in stile Meltio Space: Translate / Rotate"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("background:#f5f5f5; padding:10px; border:1px solid #ccc;")
        content_layout.addWidget(instructions)

        tabs = QTabWidget()
        content_layout.addWidget(tabs)

        # TAB INPUT
        input_tab = QWidget()
        tabs.addTab(input_tab, "Input")

        input_layout = QVBoxLayout(input_tab)

        import_row = QHBoxLayout()
        self.import_btn = QPushButton("Import TXT")
        import_row.addWidget(self.import_btn)
        import_row.addStretch(1)
        input_layout.addLayout(import_row)

        real_group = QGroupBox("Dati reali da tastatura")
        real_layout = QVBoxLayout(real_group)
        input_layout.addWidget(real_group)

        real_split = QHBoxLayout()
        real_layout.addLayout(real_split)

        self.hole1_widget = HoleInputWidget("Foro 1 reale")
        self.hole2_widget = HoleInputWidget("Foro 2 reale")
        self.plane_widget = PlaneInputWidget("Piano reale")
        real_split.addWidget(self.hole1_widget, 1)
        real_split.addWidget(self.hole2_widget, 1)
        real_split.addWidget(self.plane_widget, 1)

        cad_group = QGroupBox("Dati nominali CAD")
        cad_layout = QVBoxLayout(cad_group)
        input_layout.addWidget(cad_group)

        cad_info = QLabel(
            "Metodo CAD:\n"
            "- Inserire centro foro 1 nominale\n"
            "- Inserire centro foro 2 nominale\n"
            "- Inserire quota del piano nominale Z\n"
            "- Il piano nominale è assunto parallelo a XY"
        )
        cad_info.setWordWrap(True)
        cad_info.setStyleSheet("color: #333;")
        cad_layout.addWidget(cad_info)

        self.nom_hole1 = XYZInputRow("Foro 1 CAD")
        self.nom_hole2 = XYZInputRow("Foro 2 CAD")
        self.nom_plane_height = QDoubleSpinBox()
        self.nom_plane_height.setRange(-1_000_000, 1_000_000)
        self.nom_plane_height.setDecimals(6)
        self.nom_plane_height.setSingleStep(0.1)
        self.nom_plane_height.setValue(0.0)

        cad_layout.addWidget(self.nom_hole1)
        cad_layout.addWidget(self.nom_hole2)

        zrow = QHBoxLayout()
        zrow.addWidget(QLabel("Quota piano nominale CAD (Z = h)"))
        zrow.addWidget(self.nom_plane_height)
        zrow.addStretch(1)
        cad_layout.addLayout(zrow)

        opt_group = QGroupBox("Opzioni")
        opt_layout = QHBoxLayout(opt_group)
        input_layout.addWidget(opt_group)

        self.flip_real_z = QCheckBox("Inverti Z reale")
        self.flip_nominal_z = QCheckBox("Inverti Z nominale (usa Zn = (0,0,-1))")
        opt_layout.addWidget(self.flip_real_z)
        opt_layout.addWidget(self.flip_nominal_z)
        opt_layout.addStretch(1)

        self.thresholds = ThresholdsWidget()
        input_layout.addWidget(self.thresholds)

        # TAB RESULTS
        results_tab = QWidget()
        tabs.addTab(results_tab, "Risultati")
        results_layout = QVBoxLayout(results_tab)

        btn_row = QHBoxLayout()
        self.calc_btn = QPushButton("Calcola trasformazione")
        self.save_btn = QPushButton("Salva TXT")
        self.clear_btn = QPushButton("Pulisci output")
        btn_row.addWidget(self.calc_btn)
        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(self.clear_btn)
        btn_row.addStretch(1)
        results_layout.addLayout(btn_row)

        self.output = QTextEdit()
        self.output.setLineWrapMode(QTextEdit.NoWrap)
        self.output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results_layout.addWidget(self.output)

        self.import_btn.clicked.connect(self.import_txt)
        self.calc_btn.clicked.connect(self.calculate_all)
        self.save_btn.clicked.connect(self.save_txt)
        self.clear_btn.clicked.connect(self.output.clear)

    # ---------------------------
    # Core calculation pipeline
    # ---------------------------
    def import_txt(self):
        path, _ = QFileDialog.getOpenFileName(self, "Apri file TXT", "", "Text Files (*.txt)")
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.parse_txt_lines(lines)
        except Exception as e:
            QMessageBox.critical(self, "Import TXT", str(e))

    def parse_txt_lines(self, lines):
        mode = None

        hole1_pts = []
        hole2_pts = []
        plane_pts = []
        hole1_center = None
        hole2_center = None

        for raw_line in lines:
            line = raw_line.strip()

            if not line:
                continue

            upper_line = line.upper()
            if line.startswith("#") or upper_line in {
                "FORO 1 - PUNTI",
                "FORO 2 - PUNTI",
                "FORO 1 - CENTRO",
                "FORO 2 - CENTRO",
                "PIANO Z",
                "CAD NOMINALE",
            }:
                if "FORO 1 - PUNTI" in upper_line:
                    mode = "hole1_pts"
                elif "FORO 2 - PUNTI" in upper_line:
                    mode = "hole2_pts"
                elif "FORO 1 - CENTRO" in upper_line:
                    mode = "hole1_center"
                elif "FORO 2 - CENTRO" in upper_line:
                    mode = "hole2_center"
                elif "PIANO Z" in upper_line:
                    mode = "plane"
                elif "CAD NOMINALE" in upper_line:
                    mode = "cad"
                continue

            if "=" in line:
                key, val = line.split("=", 1)
                nums = [parse_float_input(x) for x in val.strip().split()]
                if key.strip().upper() == "F1":
                    self.nom_hole1.x.setValue(nums[0])
                    self.nom_hole1.y.setValue(nums[1])
                    self.nom_hole1.z.setValue(nums[2])
                elif key.strip().upper() == "F2":
                    self.nom_hole2.x.setValue(nums[0])
                    self.nom_hole2.y.setValue(nums[1])
                    self.nom_hole2.z.setValue(nums[2])
                elif key.strip().upper() == "Z":
                    self.nom_plane_height.setValue(nums[0])
                continue

            parts = line.split()

            if len(parts) == 3:
                vals = [parse_float_input(p) for p in parts]

                if mode == "hole1_pts":
                    hole1_pts.append(vals)
                elif mode == "hole2_pts":
                    hole2_pts.append(vals)
                elif mode == "plane":
                    plane_pts.append(vals)
                elif mode == "hole1_center":
                    hole1_center = vals
                elif mode == "hole2_center":
                    hole2_center = vals

        self.fill_table(self.plane_widget.table, plane_pts)

        if hole1_pts:
            self.hole1_widget.mode.setCurrentIndex(0)
            self.fill_table(self.hole1_widget.points_table, hole1_pts)
        elif hole1_center:
            self.hole1_widget.mode.setCurrentIndex(1)
            self.set_center(self.hole1_widget.center_row, hole1_center)

        if hole2_pts:
            self.hole2_widget.mode.setCurrentIndex(0)
            self.fill_table(self.hole2_widget.points_table, hole2_pts)
        elif hole2_center:
            self.hole2_widget.mode.setCurrentIndex(1)
            self.set_center(self.hole2_widget.center_row, hole2_center)

    def fill_table(self, table, points):
        table.setRowCount(len(points))
        for r, p in enumerate(points):
            for c in range(3):
                table.setItem(r, c, QTableWidgetItem(str(p[c])))

    def set_center(self, widget, vals):
        widget.x.setValue(vals[0])
        widget.y.setValue(vals[1])
        widget.z.setValue(vals[2])

    def calculate_all(self):
        try:
            hole1 = self.hole1_widget.get_result()
            hole2 = self.hole2_widget.get_result()
            plane = self.plane_widget.get_result()

            Zr = orient_real_plane_normal(
                plane.normal,
                force_flip=self.flip_real_z.isChecked()
            )

            # Proiezioni reali
            F1r_proj = project_point_on_plane(hole1.center_raw, plane.point, Zr)
            F2r_proj = project_point_on_plane(hole2.center_raw, plane.point, Zr)

            real_frame = build_frame_from_holes_and_plane(F1r_proj, F2r_proj, Zr)

            # CAD nominale
            F1n = self.nom_hole1.value()
            F2n = self.nom_hole2.value()
            h = self.nom_plane_height.value()

            Zn = np.array([0.0, 0.0, -1.0 if self.flip_nominal_z.isChecked() else 1.0])

            # Proiezione nominale lungo Zn sul piano z=h
            # con piano parallelo a XY, il risultato è semplicemente sostituire la coordinata z con h
            F1n_proj = np.array([F1n[0], F1n[1], h], dtype=float)
            F2n_proj = np.array([F2n[0], F2n[1], h], dtype=float)

            nominal_frame = build_frame_from_holes_and_plane(F1n_proj, F2n_proj, Zn)

            # Trasformazione nominale -> reale
            R = real_frame.R @ nominal_frame.R.T
            t = real_frame.origin - R @ nominal_frame.origin
            T = homogeneous_from_rt(R, t)

            rz, ry, rx = rotation_matrix_to_euler_zyx_deg(R)

            quality = build_quality_report(
                hole1=hole1,
                hole2=hole2,
                plane=plane,
                F1r=F1r_proj,
                F2r=F2r_proj,
                F1n_proj=F1n_proj,
                F2n_proj=F2n_proj,
                Zr=real_frame.Z,
                Xr=real_frame.X,
                thresholds=self.thresholds.values()
            )

            report = self.build_report(
                hole1, hole2, plane,
                F1r_proj, F2r_proj,
                F1n_proj, F2n_proj,
                real_frame, nominal_frame,
                R, t, T,
                rz, ry, rx,
                quality
            )
            self.output.setPlainText(report)
            self.colorize_output_status(quality.status)

        except Exception as e:
            msg = "ERRORE DI CALCOLO\n\n" + str(e) + "\n\n" + traceback.format_exc()
            self.output.setPlainText(msg)
            self.colorize_output_status("CRITICAL")

    def build_report(
        self,
        hole1: HoleInputResult,
        hole2: HoleInputResult,
        plane: PlaneFitResult,
        F1r_proj: np.ndarray,
        F2r_proj: np.ndarray,
        F1n_proj: np.ndarray,
        F2n_proj: np.ndarray,
        real_frame: FrameResult,
        nominal_frame: FrameResult,
        R: np.ndarray,
        t: np.ndarray,
        T: np.ndarray,
        rz: float, ry: float, rx: float,
        quality: QualityReport
    ) -> str:
        lines = []

        lines.append("OUTPUT FOR MELTIO SPACE")
        lines.append("")
        lines.append("PART TRANSFORM")
        lines.append("")
        lines.append("TRANSLATE")
        lines.append(f"X = {t[0]:.6f}")
        lines.append(f"Y = {t[1]:.6f}")
        lines.append(f"Z = {t[2]:.6f}")
        lines.append("")
        lines.append("ROTATE")
        lines.append(f"Z = {rz:.6f}")
        lines.append(f"Y = {ry:.6f}")
        lines.append(f"X = {rx:.6f}")
        lines.append("")

        lines.append("REAL DATA")
        lines.append(f"Foro 1 raw = {format_vec(hole1.center_raw)}")
        lines.append(f"Foro 2 raw = {format_vec(hole2.center_raw)}")
        lines.append(f"Foro 1 projected = {format_vec(F1r_proj)}")
        lines.append(f"Foro 2 projected = {format_vec(F2r_proj)}")
        lines.append(f"Piano reale point = {format_vec(plane.point)}")
        lines.append(f"Piano reale normal = {format_vec(real_frame.Z)}")
        lines.append("")

        lines.append("CAD NOMINAL DATA")
        lines.append(f"Foro 1 nominal projected = {format_vec(F1n_proj)}")
        lines.append(f"Foro 2 nominal projected = {format_vec(F2n_proj)}")
        lines.append(f"Nominal Z axis = {format_vec(nominal_frame.Z)}")
        lines.append("")

        lines.append("REAL FRAME AXES")
        lines.append(f"X_dir = {format_vec(real_frame.X)}")
        lines.append(f"Y_dir = {format_vec(real_frame.Y)}")
        lines.append(f"Z_dir = {format_vec(real_frame.Z)}")
        lines.append("")

        lines.append("NOMINAL FRAME AXES")
        lines.append(f"Xn_dir = {format_vec(nominal_frame.X)}")
        lines.append(f"Yn_dir = {format_vec(nominal_frame.Y)}")
        lines.append(f"Zn_dir = {format_vec(nominal_frame.Z)}")
        lines.append("")

        lines.append("ROTATION MATRIX R")
        lines.append(format_matrix(R))
        lines.append("")
        lines.append("HOMOGENEOUS MATRIX T")
        lines.append(format_matrix(T))
        lines.append("")

        lines.append("FIT DETAILS")
        lines.append(f"Piano reale RMS = {plane.rms:.6f}")
        lines.append(f"Piano reale Max residual = {plane.max_residual:.6f}")
        lines.append(f"Piano reale Area indicator = {plane.area_indicator:.6f}")

        for idx, hole in [(1, hole1), (2, hole2)]:
            if hole.source == "points" and hole.circle_fit is not None:
                cf = hole.circle_fit
                lines.append(
                    f"Foro {idx}: fit da punti, center={format_vec(cf.center_3d)}, "
                    f"radius={cf.radius:.6f}, RMS={cf.rms:.6f}, Max={cf.max_residual:.6f}"
                )
            else:
                lines.append(f"Foro {idx}: centro inserito direttamente")

        lines.append("")
        lines.append(f"QUALITY STATUS = {quality.status}")
        for line in quality.lines:
            lines.append(line)

        return "\n".join(lines)

    def colorize_output_status(self, status: str):
        if status == "OK":
            color = "#e9f9e9"
        elif status == "WARNING":
            color = "#fff7dd"
        else:
            color = "#ffe5e5"
        self.output.setStyleSheet(f"background:{color}; font-family: Consolas, monospace; font-size: 12px;")

    def save_txt(self):
        content = self.output.toPlainText().strip()
        if not content:
            QMessageBox.warning(self, "Salvataggio", "Nessun contenuto da salvare.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Salva output",
            "",
            "Text Files (*.txt)"
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            QMessageBox.information(self, "Salvataggio", "File salvato correttamente.")
        except Exception as e:
            QMessageBox.critical(self, "Errore salvataggio", str(e))


# ============================================================
# MAIN
# ============================================================

def main():
    app = QApplication(sys.argv)
    win = MeltioFrameTool()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
