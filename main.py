import sys
import math
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QTextCursor, QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QTextEdit, QFileDialog, QTableWidget,
    QTableWidgetItem, QGroupBox, QComboBox, QAbstractSpinBox, QDoubleSpinBox,
    QMessageBox, QCheckBox, QTabWidget, QSizePolicy, QScrollArea, QStackedWidget
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
class CircleDatumInputResult:
    center_raw: np.ndarray
    source: str  # "points" or "center"
    circle_fit: Optional[CircleFitResult]
    imported_center_ignored: bool = False


@dataclass
class LineFitResult:
    point: np.ndarray
    direction: np.ndarray
    rms: float
    max_residual: float
    num_points: int


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
ASSET_DIR = Path(__file__).resolve().parent


def asset_path(filename: str) -> Path:
    return ASSET_DIR / filename


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


def fit_circle_on_plane(points: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> CircleFitResult:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        raise ValueError("Servono almeno 3 punti per il fit del cerchio.")

    proj_pts, pts_2d, u, v = project_points_to_plane_basis(pts, plane_point, plane_normal)
    center_2d, radius, residuals = fit_circle_2d(pts_2d)
    center_3d = plane_point + center_2d[0] * u + center_2d[1] * v

    plane_distances = (pts - proj_pts) @ normalize(plane_normal)
    total_residuals = np.sqrt(residuals ** 2 + plane_distances ** 2)

    return CircleFitResult(
        center_3d=center_3d,
        radius=radius,
        rms=float(np.sqrt(np.mean(total_residuals ** 2))),
        max_residual=float(np.max(np.abs(total_residuals))),
        local_plane_normal=normalize(plane_normal),
        local_plane_point=plane_point,
        num_points=len(points),
    )


def fit_line(points: np.ndarray) -> LineFitResult:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        raise ValueError("Servono almeno 2 punti per la linea.")

    centroid = pts.mean(axis=0)
    if len(pts) == 2:
        direction = normalize(pts[1] - pts[0])
    else:
        A = pts - centroid
        _, _, vh = np.linalg.svd(A, full_matrices=False)
        direction = normalize(vh[0])

    rel = pts - centroid
    cross_dist = np.linalg.norm(np.cross(rel, direction), axis=1)

    return LineFitResult(
        point=centroid,
        direction=direction,
        rms=float(np.sqrt(np.mean(cross_dist ** 2))),
        max_residual=float(np.max(cross_dist)),
        num_points=len(points),
    )


def project_direction_to_plane(direction: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    n = normalize(plane_normal)
    d = np.asarray(direction, dtype=float)
    projected = d - np.dot(d, n) * n
    return normalize(projected)


def build_frame_from_origin_x_and_z(origin: np.ndarray, x_direction: np.ndarray, z_direction: np.ndarray) -> FrameResult:
    Z = normalize(z_direction)
    X = project_direction_to_plane(x_direction, Z)

    cross_mag = np.linalg.norm(np.cross(Z, X))
    if cross_mag < 1e-6:
        raise ValueError("Asse X quasi parallelo a Z: frame non stabile.")

    Y = normalize(np.cross(Z, X))
    X = normalize(np.cross(Y, Z))

    X, Y, Z = ensure_right_handed(X, Y, Z)
    R = np.column_stack((X, Y, Z))

    return FrameResult(origin=origin, R=R, X=X, Y=Y, Z=Z)


def nominal_axis_from_key(key: str) -> np.ndarray:
    axes = {
        "+x": np.array([1.0, 0.0, 0.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "+y": np.array([0.0, 1.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
    }
    if key not in axes:
        raise ValueError("Direzione CAD Datum C non valida.")
    return axes[key]


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


def rotation_matrix_to_euler_xyz_deg(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Restituisce angoli XYZ in gradi: (Rx, Ry, Rz)
    """
    sy = max(-1.0, min(1.0, R[0, 2]))
    ry = math.asin(sy)
    cy = math.cos(ry)

    if abs(cy) > 1e-8:
        rx = math.atan2(-R[1, 2], R[2, 2])
        rz = math.atan2(-R[0, 1], R[0, 0])
    else:
        rx = math.atan2(R[2, 1], R[1, 1])
        rz = 0.0

    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def rotation_matrix_to_euler_xzy_deg(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Restituisce angoli XZY in gradi: (Rx, Rz, Ry)
    """
    sz = max(-1.0, min(1.0, -R[0, 1]))
    rz = math.asin(sz)
    cz = math.cos(rz)

    if abs(cz) > 1e-8:
        rx = math.atan2(R[2, 1], R[1, 1])
        ry = math.atan2(R[0, 2], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[2, 2])
        ry = 0.0

    return math.degrees(rx), math.degrees(rz), math.degrees(ry)


def rotation_matrix_to_euler_zxy_deg(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Restituisce angoli ZXY in gradi: (Rz, Rx, Ry)
    """
    sx = max(-1.0, min(1.0, R[2, 1]))
    rx = math.asin(sx)
    cx = math.cos(rx)

    if abs(cx) > 1e-8:
        ry = math.atan2(-R[2, 0], R[2, 2])
        rz = math.atan2(-R[0, 1], R[1, 1])
    else:
        ry = math.atan2(R[0, 2], R[0, 0])
        rz = 0.0

    return math.degrees(rz), math.degrees(rx), math.degrees(ry)


def rotation_matrix_to_euler_yxz_deg(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Restituisce angoli YXZ in gradi: (Ry, Rx, Rz)
    """
    sx = max(-1.0, min(1.0, -R[1, 2]))
    rx = math.asin(sx)
    cx = math.cos(rx)

    if abs(cx) > 1e-8:
        ry = math.atan2(R[0, 2], R[2, 2])
        rz = math.atan2(R[1, 0], R[1, 1])
    else:
        ry = math.atan2(-R[2, 0], R[0, 0])
        rz = 0.0

    return math.degrees(ry), math.degrees(rx), math.degrees(rz)


def rotation_matrix_to_euler_yzx_deg(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Restituisce angoli YZX in gradi: (Ry, Rz, Rx)
    """
    sz = max(-1.0, min(1.0, R[1, 0]))
    rz = math.asin(sz)
    cz = math.cos(rz)

    if abs(cz) > 1e-8:
        ry = math.atan2(-R[2, 0], R[0, 0])
        rx = math.atan2(-R[1, 2], R[1, 1])
    else:
        sign = 1.0 if sz >= 0.0 else -1.0
        ry = math.atan2(sign * R[2, 1], -sign * R[0, 1])
        rx = 0.0

    return math.degrees(ry), math.degrees(rz), math.degrees(rx)


def build_rotation_output(R: np.ndarray, mode: str) -> Tuple[str, List[Tuple[str, float]]]:
    if mode == "xyz":
        rx, ry, rz = rotation_matrix_to_euler_xyz_deg(R)
        return "XYZ", [("X", rx), ("Y", ry), ("Z", rz)]

    if mode == "swap_xz":
        rz, ry, rx = rotation_matrix_to_euler_zyx_deg(R)
        return "SWAP_XZ_OUTPUT", [("Z", rx), ("Y", ry), ("X", rz)]

    if mode == "xzy":
        rx, rz, ry = rotation_matrix_to_euler_xzy_deg(R)
        return "XZY", [("X", rx), ("Z", rz), ("Y", ry)]

    if mode == "zxy":
        rz, rx, ry = rotation_matrix_to_euler_zxy_deg(R)
        return "ZXY", [("Z", rz), ("X", rx), ("Y", ry)]

    if mode == "yxz":
        ry, rx, rz = rotation_matrix_to_euler_yxz_deg(R)
        return "YXZ", [("Y", ry), ("X", rx), ("Z", rz)]

    if mode == "yzx":
        ry, rz, rx = rotation_matrix_to_euler_yzx_deg(R)
        return "YZX", [("Y", ry), ("Z", rz), ("X", rx)]

    rz, ry, rx = rotation_matrix_to_euler_zyx_deg(R)
    return "ZYX", [("Z", rz), ("Y", ry), ("X", rx)]


def build_angle_output_warning(rotation_mode_label: str, rotation_lines: List[Tuple[str, float]]) -> List[str]:
    critical_axis_by_mode = {
        "ZYX": "Y",
        "XYZ": "Y",
        "XZY": "Z",
        "ZXY": "X",
        "YXZ": "X",
        "YZX": "Z",
        "SWAP_XZ_OUTPUT": "Y",
    }
    critical_axis = critical_axis_by_mode.get(rotation_mode_label)
    if critical_axis is None:
        return []

    angles = dict(rotation_lines)
    angle = angles.get(critical_axis)
    if angle is None:
        return []

    distance_from_gimbal = min(abs(angle - 90.0), abs(angle + 90.0))
    if distance_from_gimbal > 2.0:
        return []

    title = "AVVISO FORTE ANGOLI ROTATE" if distance_from_gimbal <= 0.5 else "AVVISO ANGOLI ROTATE"
    if rotation_mode_label == "SWAP_XZ_OUTPUT":
        detail = (
            f"Swap X/Z è una rimappatura speciale basata su ZYX: "
            f"angolo Y = {angle:.6f}, vicino al gimbal lock."
        )
    else:
        detail = (
            f"Modalità {rotation_mode_label} vicina al gimbal lock: "
            f"angolo {critical_axis} = {angle:.6f}."
        )

    return [
        title,
        detail,
        "Riguarda solo i tre valori Rotate: matrice R e Translate restano validi.",
    ]


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


class ManualDoubleSpinBox(QDoubleSpinBox):
    def __init__(self):
        super().__init__()
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

    def wheelEvent(self, event):
        event.ignore()

    def stepBy(self, steps: int):
        return


class XYZInputRow(QWidget):
    def __init__(self, title: str):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel(title))

        self.x = ManualDoubleSpinBox()
        self.y = ManualDoubleSpinBox()
        self.z = ManualDoubleSpinBox()
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


class CircleDatumInputWidget(QGroupBox):
    def __init__(self, title: str, instructions: str, rows: int = 8):
        super().__init__(title)
        layout = QVBoxLayout(self)

        self.imported_center_ignored = False

        self.mode = QComboBox()
        self.mode.addItems([
            "Punti tastati",
            "Centro già calcolato"
        ])
        layout.addWidget(QLabel("Modalità input"))
        layout.addWidget(self.mode)

        info = QLabel(instructions)
        info.setWordWrap(True)
        info.setStyleSheet("color: #333;")
        layout.addWidget(info)

        self.points_table = create_empty_table(rows)
        layout.addWidget(self.points_table)

        btn_row = QHBoxLayout()
        self.add_row_btn = QPushButton("Aggiungi riga punti")
        self.remove_row_btn = QPushButton("Rimuovi riga punti")
        btn_row.addWidget(self.add_row_btn)
        btn_row.addWidget(self.remove_row_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.center_row = XYZInputRow("Centro")
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

    def set_points(self, points):
        self.points_table.setRowCount(len(points))
        for r, p in enumerate(points):
            for c in range(3):
                self.points_table.setItem(r, c, QTableWidgetItem(str(p[c])))

    def set_center(self, vals):
        self.center_row.x.setValue(vals[0])
        self.center_row.y.setValue(vals[1])
        self.center_row.z.setValue(vals[2])

    def get_result(self, plane_point: np.ndarray, plane_normal: np.ndarray) -> CircleDatumInputResult:
        if self.mode.currentIndex() == 0:
            pts = read_points_from_table(self.points_table)
            if len(pts) < 3:
                raise ValueError(f"{self.title()}: servono almeno 3 punti.")
            circle = fit_circle_on_plane(pts, plane_point, plane_normal)
            return CircleDatumInputResult(
                center_raw=circle.center_3d,
                source="points",
                circle_fit=circle,
                imported_center_ignored=self.imported_center_ignored
            )

        center = self.center_row.value()
        return CircleDatumInputResult(
            center_raw=center,
            source="center",
            circle_fit=None
        )


class LineInputWidget(QGroupBox):
    def __init__(self, title: str):
        super().__init__(title)
        layout = QVBoxLayout(self)

        info = QLabel(
            "Metodo linea / asse:\n"
            "- Inserire almeno 2 punti sul riferimento lineare.\n"
            "- Con 2 punti viene usata la direzione punto 1 -> punto 2.\n"
            "- Con più punti viene calcolata la linea media.\n"
            "- La direzione viene proiettata sul piano superiore."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #333;")
        layout.addWidget(info)

        self.table = create_empty_table(4)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.add_row_btn = QPushButton("Aggiungi riga linea")
        self.remove_row_btn = QPushButton("Rimuovi riga linea")
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

    def get_result(self) -> LineFitResult:
        pts = read_points_from_table(self.table)
        return fit_line(pts)


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
            ("plane_rms_warning", "Errore piano warning", 0.05, "Scostamento medio dei punti dal piano calcolato."),
            ("plane_rms_critical", "Errore piano critical", 0.15, "Limite critico dello scostamento medio dal piano."),
            ("plane_area_warning", "Area piano warning", 1.0, "Estensione dell'area coperta dai punti tastati sul piano."),
            ("hole_rms_warning", "Errore foro warning", 0.05, "Scostamento medio dei punti dal cerchio calcolato."),
            ("hole_rms_critical", "Errore foro critical", 0.15, "Limite critico dello scostamento medio dal cerchio."),
            ("hole_distance_critical", "Interasse minimo critical", 1.0, "Distanza minima ammessa tra foro 1 e foro 2."),
            ("distance_delta_warning", "Delta interasse warning", 0.20, "Differenza tra distanza CAD e distanza reale dei fori."),
            ("distance_delta_critical", "Delta interasse critical", 0.50, "Limite critico della differenza CAD/reale dei fori."),
            ("xz_cross_warning", "Ortogonalità assi (|Z x X|) warning", 0.10, "Indice di ortogonalità tra asse Z del piano e asse X dei fori."),
            ("xz_cross_critical", "Ortogonalità assi (|Z x X|) critical", 0.01, "Limite critico dell'ortogonalità tra asse Z e asse X."),
        ]

        for r, (key, label, default, help_text) in enumerate(rows):
            sb = ManualDoubleSpinBox()
            sb.setRange(0.0, 1_000_000.0)
            sb.setDecimals(6)
            sb.setValue(default)
            sb.setSingleStep(0.01)
            help_label = QLabel(help_text)
            help_label.setWordWrap(True)
            help_label.setStyleSheet("color: #666; font-size: 10px;")
            self.widgets[key] = sb
            layout.addWidget(QLabel(label), r, 0)
            layout.addWidget(sb, r, 1)
            layout.addWidget(help_label, r, 2)
        layout.setColumnStretch(2, 1)

    def values(self) -> dict:
        return {k: w.value() for k, w in self.widgets.items()}


# ============================================================
# MAIN WINDOW
# ============================================================

class MeltioFrameTool(QWidget):
    def __init__(self):
        super().__init__()
        window_icon = QIcon(str(asset_path("logo777.ico")))
        if not window_icon.isNull():
            self.setWindowIcon(window_icon)
        self.setWindowTitle("Tool di centraggio reale -> Meltio Space")
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

        header_logo_size = 72
        title_row = QHBoxLayout()
        logo_label = QLabel()
        logo_path = asset_path("logo777_black on transparent.png")
        logo_pixmap = QPixmap(str(logo_path))
        if not logo_pixmap.isNull():
            logo_label.setPixmap(
                logo_pixmap.scaled(header_logo_size, header_logo_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        title = QLabel("Tool -BASIC- per centraggio CAD → reale")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        wire_logo_label = QLabel()
        wire_logo_path = asset_path("Wire-trading.png")
        wire_logo_pixmap = QPixmap(str(wire_logo_path))
        if not wire_logo_pixmap.isNull():
            wire_logo_label.setPixmap(
                wire_logo_pixmap.scaled(header_logo_size, header_logo_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        title_row.addWidget(logo_label)
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(wire_logo_label)
        content_layout.addLayout(title_row)

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
        self.nom_plane_height = ManualDoubleSpinBox()
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

        plane_comp_group = QGroupBox("Compensazione piano tastato")
        plane_comp_layout = QGridLayout(plane_comp_group)
        input_layout.addWidget(plane_comp_group)

        plane_comp_info = QLabel(
            "Applica la compensazione del raggio sfera solo al piano reale.\n"
            "I fori non vengono compensati: il centro resta corretto, cambia solo il raggio misurato.\n"
            "Scegli il verso in cui si trova il piano reale rispetto a Z reale."
        )
        plane_comp_info.setWordWrap(True)
        plane_comp_info.setStyleSheet("color: #333;")
        plane_comp_layout.addWidget(plane_comp_info, 0, 0, 1, 2)

        self.plane_comp_mode = QComboBox()
        self.plane_comp_mode.addItems([
            "Nessuna",
            "Piano reale nel verso di Z reale",
            "Piano reale nel verso opposto a Z reale",
        ])
        plane_comp_layout.addWidget(QLabel("Compensazione"), 1, 0)
        plane_comp_layout.addWidget(self.plane_comp_mode, 1, 1)

        self.probe_sphere_diameter = ManualDoubleSpinBox()
        self.probe_sphere_diameter.setRange(0.0, 1_000_000.0)
        self.probe_sphere_diameter.setDecimals(6)
        self.probe_sphere_diameter.setSingleStep(0.1)
        self.probe_sphere_diameter.setValue(0.0)
        plane_comp_layout.addWidget(QLabel("Diametro sfera"), 2, 0)
        plane_comp_layout.addWidget(self.probe_sphere_diameter, 2, 1)

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

        rotation_output_group = QGroupBox("Output rotazioni")
        rotation_output_layout = QGridLayout(rotation_output_group)
        results_layout.addWidget(rotation_output_group)

        rotation_mode_info = QLabel(
            "Modalità output: sceglie come scrivere la stessa rotazione nei campi Rotate X/Y/Z. "
            "Matrice R e Translate non cambiano."
        )
        rotation_mode_info.setWordWrap(True)
        rotation_mode_info.setStyleSheet("color: #333;")
        rotation_output_layout.addWidget(rotation_mode_info, 0, 0, 1, 2)

        gimbal_info = QLabel(
            "Gimbal lock: in alcune posizioni i tre angoli Rotate possono diventare ambigui. "
            "Il report avvisa solo quando la modalità scelta è vicina al problema."
        )
        gimbal_info.setWordWrap(True)
        gimbal_info.setStyleSheet("color: #555; font-size: 10px;")
        rotation_output_layout.addWidget(gimbal_info, 1, 0, 1, 2)

        swap_info = QLabel(
            "Swap X/Z: opzione speciale per rimappare i campi rotazione X e Z; "
            "non è una convenzione Euler equivalente."
        )
        swap_info.setWordWrap(True)
        swap_info.setStyleSheet("color: #555; font-size: 10px;")
        rotation_output_layout.addWidget(swap_info, 2, 0, 1, 2)

        self.rotation_output_mode = QComboBox()
        self.rotation_output_mode.addItem("ZYX (attuale)", "zyx")
        self.rotation_output_mode.addItem("XYZ", "xyz")
        self.rotation_output_mode.addItem("Swap X/Z output", "swap_xz")
        self.rotation_output_mode.addItem("XZY (avanzata)", "xzy")
        self.rotation_output_mode.addItem("ZXY (avanzata)", "zxy")
        self.rotation_output_mode.addItem("YXZ (avanzata, poco probabile in Space)", "yxz")
        self.rotation_output_mode.addItem("YZX (avanzata, poco probabile in Space)", "yzx")
        rotation_output_layout.addWidget(QLabel("Convenzione output"), 3, 0)
        rotation_output_layout.addWidget(self.rotation_output_mode, 3, 1)

        self.output = QTextEdit()
        self.output.setLineWrapMode(QTextEdit.NoWrap)
        self.output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results_layout.addWidget(self.output)

        self.import_btn.clicked.connect(self.import_txt)
        self.calc_btn.clicked.connect(self.calculate_all)
        self.save_btn.clicked.connect(self.save_txt)
        self.clear_btn.clicked.connect(self.output.clear)

        footer = QLabel("This software is licensed by SiRe, VAT No. IT01314390251, for use by Wire Trading.")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #666; font-size: 9px;")
        main_layout.addWidget(footer)

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

    def get_plane_compensation(self, plane_normal: np.ndarray) -> Tuple[np.ndarray, float, float, int]:
        diameter = self.probe_sphere_diameter.value()
        radius = diameter / 2.0
        sign_by_mode = {0: 0, 1: 1, 2: -1}
        sign = sign_by_mode.get(self.plane_comp_mode.currentIndex(), 0)

        if diameter <= 0.0 or sign == 0:
            return np.zeros(3, dtype=float), diameter, radius, 0

        return sign * radius * normalize(plane_normal), diameter, radius, sign

    def get_rotation_output_mode(self) -> str:
        return self.rotation_output_mode.currentData() or "zyx"

    def calculate_all(self):
        try:
            hole1 = self.hole1_widget.get_result()
            hole2 = self.hole2_widget.get_result()
            plane = self.plane_widget.get_result()

            Zr = orient_real_plane_normal(
                plane.normal,
                force_flip=self.flip_real_z.isChecked()
            )
            plane_comp_vector, plane_comp_diameter, plane_comp_radius, plane_comp_sign = self.get_plane_compensation(Zr)
            plane_point_used = plane.point + plane_comp_vector

            # Proiezioni reali
            F1r_proj = project_point_on_plane(hole1.center_raw, plane_point_used, Zr)
            F2r_proj = project_point_on_plane(hole2.center_raw, plane_point_used, Zr)

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
            rotation_mode_label, rotation_lines = build_rotation_output(R, self.get_rotation_output_mode())
            angle_output_warning = build_angle_output_warning(rotation_mode_label, rotation_lines)

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
                plane_point_used, plane_comp_vector,
                plane_comp_diameter, plane_comp_radius, plane_comp_sign,
                F1r_proj, F2r_proj,
                F1n_proj, F2n_proj,
                real_frame, nominal_frame,
                R, t, T,
                rotation_mode_label, rotation_lines,
                angle_output_warning,
                quality
            )
            self.output.setPlainText(report)
            self.colorize_output_status(quality.status)

        except ValueError as e:
            msg = "ERRORE DI INPUT\n\n" + str(e)
            self.output.setPlainText(msg)
            self.colorize_output_status("CRITICAL")
        except Exception as e:
            msg = "ERRORE DI CALCOLO\n\n" + str(e) + "\n\n" + traceback.format_exc()
            self.output.setPlainText(msg)
            self.colorize_output_status("CRITICAL")

    def build_report(
        self,
        hole1: HoleInputResult,
        hole2: HoleInputResult,
        plane: PlaneFitResult,
        plane_point_used: np.ndarray,
        plane_comp_vector: np.ndarray,
        plane_comp_diameter: float,
        plane_comp_radius: float,
        plane_comp_sign: int,
        F1r_proj: np.ndarray,
        F2r_proj: np.ndarray,
        F1n_proj: np.ndarray,
        F2n_proj: np.ndarray,
        real_frame: FrameResult,
        nominal_frame: FrameResult,
        R: np.ndarray,
        t: np.ndarray,
        T: np.ndarray,
        rotation_mode_label: str,
        rotation_lines: List[Tuple[str, float]],
        angle_output_warning: List[str],
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
        lines.append(f"ROTATE ({rotation_mode_label})")
        for axis_label, angle_value in rotation_lines:
            lines.append(f"{axis_label} = {angle_value:.6f}")
        if angle_output_warning:
            lines.append("")
            lines.extend(angle_output_warning)
        lines.append("")

        lines.append("REAL DATA")
        lines.append(f"Foro 1 raw = {format_vec(hole1.center_raw)}")
        lines.append(f"Foro 2 raw = {format_vec(hole2.center_raw)}")
        lines.append(f"Foro 1 projected = {format_vec(F1r_proj)}")
        lines.append(f"Foro 2 projected = {format_vec(F2r_proj)}")
        lines.append(f"Piano reale point raw = {format_vec(plane.point)}")
        lines.append(f"Piano reale point used = {format_vec(plane_point_used)}")
        lines.append(f"Piano reale normal = {format_vec(real_frame.Z)}")
        if plane_comp_sign == 0 or plane_comp_diameter <= 0.0:
            lines.append("Compensazione piano = nessuna")
        else:
            comp_label = "+r lungo Z reale" if plane_comp_sign > 0 else "-r lungo Z reale"
            lines.append(
                f"Compensazione piano = {comp_label}, diametro sfera={plane_comp_diameter:.6f}, "
                f"raggio={plane_comp_radius:.6f}, vettore={format_vec(plane_comp_vector)}"
            )
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
# CYLINDER WINDOW
# ============================================================

class CylinderFrameTool(QWidget):
    def __init__(self):
        super().__init__()
        window_icon = QIcon(str(asset_path("logo777.ico")))
        if not window_icon.isNull():
            self.setWindowIcon(window_icon)
        self.setWindowTitle("Tool cilindro decentrato -> Meltio Space")
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

        header_logo_size = 72
        title_row = QHBoxLayout()
        logo_label = QLabel()
        logo_pixmap = QPixmap(str(asset_path("logo777_black on transparent.png")))
        if not logo_pixmap.isNull():
            logo_label.setPixmap(
                logo_pixmap.scaled(header_logo_size, header_logo_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        title = QLabel("Tool cilindro con foro decentrato")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        wire_logo_label = QLabel()
        wire_logo_pixmap = QPixmap(str(asset_path("Wire-trading.png")))
        if not wire_logo_pixmap.isNull():
            wire_logo_label.setPixmap(
                wire_logo_pixmap.scaled(header_logo_size, header_logo_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        title_row.addWidget(logo_label)
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(wire_logo_label)
        content_layout.addLayout(title_row)

        instructions = QLabel(
            "Riferimenti metrologici usati dal software:\n"
            "- Origine reale: centro del cilindro esterno sul piano superiore reale\n"
            "- Asse Z reale: normale del piano superiore tastato\n"
            "- Asse X reale: direzione centro cilindro → datum C decentrato, proiettata sul piano superiore\n"
            "- Asse Y reale: calcolato automaticamente (sistema destrorso)\n"
            "- Il cilindro esterno serve a trovare il centro, non la direzione asse\n"
            "- Output principale in stile Meltio Space: Translate / Rotate"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("background:#f5f5f5; padding:10px; border:1px solid #ccc;")
        content_layout.addWidget(instructions)

        tabs = QTabWidget()
        content_layout.addWidget(tabs)

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

        self.datum_c_kind = QComboBox()
        self.datum_c_kind.addItem("Foro decentrato / cilindro piccolo", "circle")
        self.datum_c_kind.addItem("Piano laterale", "plane")
        self.datum_c_kind.addItem("Linea / asse", "line")
        self.datum_c_kind.addItem("On demand / da definire", "ondemand")
        kind_row = QHBoxLayout()
        kind_row.addWidget(QLabel("Tipo Datum C"))
        kind_row.addWidget(self.datum_c_kind)
        kind_row.addStretch(1)
        real_layout.addLayout(kind_row)

        real_split = QHBoxLayout()
        real_layout.addLayout(real_split)

        cylinder_instructions = (
            "Metodo cilindro esterno:\n"
            "- Inserire punti tastati sulla superficie cilindrica esterna.\n"
            "- I punti servono a calcolare il centro del cilindro in sezione.\n"
            "- La direzione asse non viene ricavata dal cilindro: viene presa dalla normale del piano superiore."
        )
        datum_c_instructions = (
            "Metodo foro/cilindro decentrato:\n"
            "- Preferire punti tastati se disponibili.\n"
            "- Se sono presenti sia punti sia centro, verranno usati i punti.\n"
            "- Questo datum blocca la rotazione attorno all'asse Z."
        )

        self.cylinder_widget = CircleDatumInputWidget("Cilindro esterno reale", cylinder_instructions, rows=8)
        self.plane_widget = PlaneInputWidget("Piano superiore reale")
        self.datum_c_widget = CircleDatumInputWidget("Datum C reale", datum_c_instructions, rows=8)
        self.datum_c_plane_widget = PlaneInputWidget("Piano laterale Datum C reale")
        self.datum_c_line_widget = LineInputWidget("Linea / asse Datum C reale")
        real_split.addWidget(self.cylinder_widget, 1)
        real_split.addWidget(self.plane_widget, 1)
        real_split.addWidget(self.datum_c_widget, 1)
        real_split.addWidget(self.datum_c_plane_widget, 1)
        real_split.addWidget(self.datum_c_line_widget, 1)

        cad_group = QGroupBox("Dati nominali CAD")
        cad_layout = QVBoxLayout(cad_group)
        input_layout.addWidget(cad_group)

        cad_info = QLabel(
            "Metodo CAD:\n"
            "- Inserire centro cilindro nominale\n"
            "- Se Datum C è foro/cilindro: inserire centro foro/cilindro decentrato nominale\n"
            "- Se Datum C è piano/linea: scegliere la direzione CAD nominale (+X, -X, +Y, -Y)\n"
            "- Inserire quota del piano superiore nominale Z\n"
            "- L'asse cilindro nominale è assunto normale al piano Z nominale"
        )
        cad_info.setWordWrap(True)
        cad_info.setStyleSheet("color: #333;")
        cad_layout.addWidget(cad_info)

        self.nom_cylinder_center = XYZInputRow("Centro cilindro CAD")
        self.nom_datum_c_center = XYZInputRow("Centro datum C CAD")
        self.nom_datum_c_direction_row = QWidget()
        direction_layout = QHBoxLayout(self.nom_datum_c_direction_row)
        direction_layout.setContentsMargins(0, 0, 0, 0)
        direction_layout.addWidget(QLabel("Direzione CAD Datum C"))
        self.nom_datum_c_direction = QComboBox()
        self.nom_datum_c_direction.addItem("+X CAD", "+x")
        self.nom_datum_c_direction.addItem("-X CAD", "-x")
        self.nom_datum_c_direction.addItem("+Y CAD", "+y")
        self.nom_datum_c_direction.addItem("-Y CAD", "-y")
        direction_layout.addWidget(self.nom_datum_c_direction)
        direction_layout.addStretch(1)
        self.nom_plane_height = ManualDoubleSpinBox()
        self.nom_plane_height.setRange(-1_000_000, 1_000_000)
        self.nom_plane_height.setDecimals(6)
        self.nom_plane_height.setSingleStep(0.1)
        self.nom_plane_height.setValue(0.0)

        cad_layout.addWidget(self.nom_cylinder_center)
        cad_layout.addWidget(self.nom_datum_c_center)
        cad_layout.addWidget(self.nom_datum_c_direction_row)

        zrow = QHBoxLayout()
        zrow.addWidget(QLabel("Quota piano superiore CAD (Z = h)"))
        zrow.addWidget(self.nom_plane_height)
        zrow.addStretch(1)
        cad_layout.addLayout(zrow)

        opt_group = QGroupBox("Opzioni")
        opt_layout = QHBoxLayout(opt_group)
        input_layout.addWidget(opt_group)

        self.flip_real_z = QCheckBox("Inverti Z reale")
        self.flip_nominal_z = QCheckBox("Inverti Z nominale (usa Zn = (0,0,-1))")
        self.invert_datum_c_direction = QCheckBox("Inverti direzione Datum C reale")
        opt_layout.addWidget(self.flip_real_z)
        opt_layout.addWidget(self.flip_nominal_z)
        opt_layout.addWidget(self.invert_datum_c_direction)
        opt_layout.addStretch(1)

        plane_comp_group = QGroupBox("Compensazione piano tastato")
        plane_comp_layout = QGridLayout(plane_comp_group)
        input_layout.addWidget(plane_comp_group)

        plane_comp_info = QLabel(
            "Applica la compensazione del raggio sfera solo al piano superiore reale.\n"
            "Per cilindro esterno e datum C si mantiene la stessa logica del modulo base."
        )
        plane_comp_info.setWordWrap(True)
        plane_comp_info.setStyleSheet("color: #333;")
        plane_comp_layout.addWidget(plane_comp_info, 0, 0, 1, 2)

        self.plane_comp_mode = QComboBox()
        self.plane_comp_mode.addItems([
            "Nessuna",
            "Piano reale nel verso di Z reale",
            "Piano reale nel verso opposto a Z reale",
        ])
        plane_comp_layout.addWidget(QLabel("Compensazione"), 1, 0)
        plane_comp_layout.addWidget(self.plane_comp_mode, 1, 1)

        self.probe_sphere_diameter = ManualDoubleSpinBox()
        self.probe_sphere_diameter.setRange(0.0, 1_000_000.0)
        self.probe_sphere_diameter.setDecimals(6)
        self.probe_sphere_diameter.setSingleStep(0.1)
        self.probe_sphere_diameter.setValue(0.0)
        plane_comp_layout.addWidget(QLabel("Diametro sfera"), 2, 0)
        plane_comp_layout.addWidget(self.probe_sphere_diameter, 2, 1)

        self.thresholds = ThresholdsWidget()
        input_layout.addWidget(self.thresholds)

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

        rotation_output_group = QGroupBox("Output rotazioni")
        rotation_output_layout = QGridLayout(rotation_output_group)
        results_layout.addWidget(rotation_output_group)

        rotation_mode_info = QLabel(
            "Modalità output: sceglie come scrivere la stessa rotazione nei campi Rotate X/Y/Z. "
            "Matrice R e Translate non cambiano."
        )
        rotation_mode_info.setWordWrap(True)
        rotation_mode_info.setStyleSheet("color: #333;")
        rotation_output_layout.addWidget(rotation_mode_info, 0, 0, 1, 2)

        self.rotation_output_mode = QComboBox()
        self.rotation_output_mode.addItem("ZYX (attuale)", "zyx")
        self.rotation_output_mode.addItem("XYZ", "xyz")
        self.rotation_output_mode.addItem("Swap X/Z output", "swap_xz")
        self.rotation_output_mode.addItem("XZY (avanzata)", "xzy")
        self.rotation_output_mode.addItem("ZXY (avanzata)", "zxy")
        self.rotation_output_mode.addItem("YXZ (avanzata, poco probabile in Space)", "yxz")
        self.rotation_output_mode.addItem("YZX (avanzata, poco probabile in Space)", "yzx")
        rotation_output_layout.addWidget(QLabel("Convenzione output"), 1, 0)
        rotation_output_layout.addWidget(self.rotation_output_mode, 1, 1)

        self.output = QTextEdit()
        self.output.setLineWrapMode(QTextEdit.NoWrap)
        self.output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results_layout.addWidget(self.output)

        self.import_btn.clicked.connect(self.import_txt)
        self.calc_btn.clicked.connect(self.calculate_all)
        self.save_btn.clicked.connect(self.save_txt)
        self.clear_btn.clicked.connect(self.output.clear)
        self.datum_c_kind.currentIndexChanged.connect(self.update_datum_c_visibility)
        self.update_datum_c_visibility()

    def update_datum_c_visibility(self):
        mode = self.datum_c_kind.currentData()
        self.datum_c_widget.setVisible(mode == "circle")
        self.datum_c_plane_widget.setVisible(mode == "plane")
        self.datum_c_line_widget.setVisible(mode == "line")
        self.nom_datum_c_center.setVisible(mode == "circle")
        self.nom_datum_c_direction_row.setVisible(mode in {"plane", "line"})
        self.invert_datum_c_direction.setVisible(mode in {"plane", "line"})

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
        cylinder_pts = []
        cylinder_center = None
        plane_pts = []
        datum_pts = []
        datum_center = None
        lateral_plane_pts = []
        line_pts = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            upper_line = line.upper()
            if line.startswith("#") or upper_line in {
                "CILINDRO ESTERNO - PUNTI",
                "CILINDRO ESTERNO - PUNTI TASTATI",
                "CILINDRO ESTERNO - CENTRO",
                "PIANO SUPERIORE",
                "PIANO SUPERIORE - PUNTI",
                "FORO DECENTRATO - PUNTI",
                "FORO DECENTRATO - PUNTI TASTATI",
                "FORO DECENTRATO - CENTRO",
                "PIANO LATERALE DATUM C - PUNTI",
                "PIANO LATERALE - PUNTI",
                "LINEA DATUM C - PUNTI",
                "LINEA - PUNTI",
                "CAD NOMINALE CILINDRO",
            }:
                if "CILINDRO ESTERNO" in upper_line and "PUNTI" in upper_line:
                    mode = "cylinder_pts"
                elif "CILINDRO ESTERNO" in upper_line and "CENTRO" in upper_line:
                    mode = "cylinder_center"
                elif "PIANO SUPERIORE" in upper_line:
                    mode = "plane"
                elif "FORO DECENTRATO" in upper_line and "PUNTI" in upper_line:
                    mode = "datum_pts"
                elif "FORO DECENTRATO" in upper_line and "CENTRO" in upper_line:
                    mode = "datum_center"
                elif "PIANO LATERALE" in upper_line and "PUNTI" in upper_line:
                    mode = "lateral_plane"
                    self.datum_c_kind.setCurrentIndex(self.datum_c_kind.findData("plane"))
                elif "LINEA" in upper_line and "PUNTI" in upper_line:
                    mode = "line"
                    self.datum_c_kind.setCurrentIndex(self.datum_c_kind.findData("line"))
                elif "CAD NOMINALE CILINDRO" in upper_line:
                    mode = "cad"
                continue

            if line.startswith("---"):
                mode = None
                continue

            if "=" in line:
                key, val = line.split("=", 1)
                key_upper = key.strip().upper()
                if key_upper in {"DIREZIONE_DATUM_C", "DIREZIONE_CAD_DATUM_C"}:
                    direction_value = val.strip().lower().replace(" ", "")
                    if direction_value in {"+x", "x"}:
                        self.nom_datum_c_direction.setCurrentIndex(self.nom_datum_c_direction.findData("+x"))
                    elif direction_value == "-x":
                        self.nom_datum_c_direction.setCurrentIndex(self.nom_datum_c_direction.findData("-x"))
                    elif direction_value in {"+y", "y"}:
                        self.nom_datum_c_direction.setCurrentIndex(self.nom_datum_c_direction.findData("+y"))
                    elif direction_value == "-y":
                        self.nom_datum_c_direction.setCurrentIndex(self.nom_datum_c_direction.findData("-y"))
                    else:
                        raise ValueError("DIREZIONE_DATUM_C deve essere +X, -X, +Y o -Y.")
                    continue

                nums = [parse_float_input(x) for x in val.replace(",", " ").split()]
                if key_upper == "CENTRO_CILINDRO":
                    self.nom_cylinder_center.x.setValue(nums[0])
                    self.nom_cylinder_center.y.setValue(nums[1])
                    self.nom_cylinder_center.z.setValue(nums[2])
                elif key_upper in {"CENTRO_FORO", "CENTRO_DATUM_C"}:
                    self.nom_datum_c_center.x.setValue(nums[0])
                    self.nom_datum_c_center.y.setValue(nums[1])
                    self.nom_datum_c_center.z.setValue(nums[2])
                elif key_upper in {"Z_PIANO", "Z"}:
                    self.nom_plane_height.setValue(nums[0])
                continue

            parts = line.split()
            if len(parts) == 3:
                vals = [parse_float_input(p) for p in parts]
                if mode == "cylinder_pts":
                    cylinder_pts.append(vals)
                elif mode == "cylinder_center":
                    cylinder_center = vals
                elif mode == "plane":
                    plane_pts.append(vals)
                elif mode == "datum_pts":
                    datum_pts.append(vals)
                elif mode == "datum_center":
                    datum_center = vals
                elif mode == "lateral_plane":
                    lateral_plane_pts.append(vals)
                elif mode == "line":
                    line_pts.append(vals)

        self.plane_widget.table.setRowCount(len(plane_pts))
        for r, p in enumerate(plane_pts):
            for c in range(3):
                self.plane_widget.table.setItem(r, c, QTableWidgetItem(str(p[c])))

        self.cylinder_widget.imported_center_ignored = False
        if cylinder_pts:
            self.cylinder_widget.mode.setCurrentIndex(0)
            self.cylinder_widget.set_points(cylinder_pts)
            if cylinder_center:
                self.cylinder_widget.set_center(cylinder_center)
                self.cylinder_widget.imported_center_ignored = True
        elif cylinder_center:
            self.cylinder_widget.mode.setCurrentIndex(1)
            self.cylinder_widget.set_center(cylinder_center)

        self.datum_c_widget.imported_center_ignored = False
        if datum_pts:
            self.datum_c_widget.mode.setCurrentIndex(0)
            self.datum_c_widget.set_points(datum_pts)
            if datum_center:
                self.datum_c_widget.set_center(datum_center)
                self.datum_c_widget.imported_center_ignored = True
        elif datum_center:
            self.datum_c_widget.mode.setCurrentIndex(1)
            self.datum_c_widget.set_center(datum_center)

        if lateral_plane_pts:
            self.datum_c_kind.setCurrentIndex(self.datum_c_kind.findData("plane"))
            self.datum_c_plane_widget.table.setRowCount(len(lateral_plane_pts))
            for r, p in enumerate(lateral_plane_pts):
                for c in range(3):
                    self.datum_c_plane_widget.table.setItem(r, c, QTableWidgetItem(str(p[c])))

        if line_pts:
            self.datum_c_kind.setCurrentIndex(self.datum_c_kind.findData("line"))
            self.datum_c_line_widget.table.setRowCount(len(line_pts))
            for r, p in enumerate(line_pts):
                for c in range(3):
                    self.datum_c_line_widget.table.setItem(r, c, QTableWidgetItem(str(p[c])))

    def get_plane_compensation(self, plane_normal: np.ndarray) -> Tuple[np.ndarray, float, float, int]:
        diameter = self.probe_sphere_diameter.value()
        radius = diameter / 2.0
        sign_by_mode = {0: 0, 1: 1, 2: -1}
        sign = sign_by_mode.get(self.plane_comp_mode.currentIndex(), 0)

        if diameter <= 0.0 or sign == 0:
            return np.zeros(3, dtype=float), diameter, radius, 0

        return sign * radius * normalize(plane_normal), diameter, radius, sign

    def get_rotation_output_mode(self) -> str:
        return self.rotation_output_mode.currentData() or "zyx"

    def calculate_all(self):
        try:
            datum_mode = self.datum_c_kind.currentData()
            plane = self.plane_widget.get_result()
            Zr = orient_real_plane_normal(
                plane.normal,
                force_flip=self.flip_real_z.isChecked()
            )
            plane_comp_vector, plane_comp_diameter, plane_comp_radius, plane_comp_sign = self.get_plane_compensation(Zr)
            plane_point_used = plane.point + plane_comp_vector

            cylinder = self.cylinder_widget.get_result(plane_point_used, Zr)
            origin_real = project_point_on_plane(cylinder.center_raw, plane_point_used, Zr)

            Cn = self.nom_cylinder_center.value()
            h = self.nom_plane_height.value()
            Zn = np.array([0.0, 0.0, -1.0 if self.flip_nominal_z.isChecked() else 1.0])
            origin_nom = np.array([Cn[0], Cn[1], h], dtype=float)

            datum_c = None
            lateral_plane = None
            line_fit = None

            if datum_mode == "circle":
                datum_c = self.datum_c_widget.get_result(plane_point_used, Zr)
                datum_real = project_point_on_plane(datum_c.center_raw, plane_point_used, Zr)
                real_frame = build_frame_from_holes_and_plane(origin_real, datum_real, Zr)

                Dn = self.nom_datum_c_center.value()
                datum_nom = np.array([Dn[0], Dn[1], h], dtype=float)
                nominal_frame = build_frame_from_holes_and_plane(origin_nom, datum_nom, Zn)

            elif datum_mode == "plane":
                lateral_plane = self.datum_c_plane_widget.get_result()
                x_real = project_direction_to_plane(lateral_plane.normal, Zr)
                if self.invert_datum_c_direction.isChecked():
                    x_real = -x_real
                real_frame = build_frame_from_origin_x_and_z(origin_real, x_real, Zr)
                datum_real = origin_real + real_frame.X

                x_nom = nominal_axis_from_key(self.nom_datum_c_direction.currentData() or "+x")
                nominal_frame = build_frame_from_origin_x_and_z(origin_nom, x_nom, Zn)
                datum_nom = origin_nom + nominal_frame.X

            elif datum_mode == "line":
                line_fit = self.datum_c_line_widget.get_result()
                x_real = project_direction_to_plane(line_fit.direction, Zr)
                if self.invert_datum_c_direction.isChecked():
                    x_real = -x_real
                real_frame = build_frame_from_origin_x_and_z(origin_real, x_real, Zr)
                datum_real = origin_real + real_frame.X

                x_nom = nominal_axis_from_key(self.nom_datum_c_direction.currentData() or "+x")
                nominal_frame = build_frame_from_origin_x_and_z(origin_nom, x_nom, Zn)
                datum_nom = origin_nom + nominal_frame.X

            else:
                raise ValueError(
                    "Questa variante Datum C richiede una procedura dedicata. "
                    "Contattare Wire Trading e SiRe per definire la strategia di tastatura."
                )

            R = real_frame.R @ nominal_frame.R.T
            t = real_frame.origin - R @ nominal_frame.origin
            T = homogeneous_from_rt(R, t)
            rotation_mode_label, rotation_lines = build_rotation_output(R, self.get_rotation_output_mode())
            angle_output_warning = build_angle_output_warning(rotation_mode_label, rotation_lines)

            quality = self.build_quality_report(
                datum_mode, cylinder, datum_c, lateral_plane, line_fit, plane,
                origin_real, datum_real,
                origin_nom, datum_nom,
                real_frame.Z, real_frame.X
            )

            report = self.build_report(
                datum_mode, cylinder, datum_c, lateral_plane, line_fit, plane,
                plane_point_used, plane_comp_vector,
                plane_comp_diameter, plane_comp_radius, plane_comp_sign,
                origin_real, datum_real,
                origin_nom, datum_nom,
                real_frame, nominal_frame,
                R, t, T,
                rotation_mode_label, rotation_lines,
                angle_output_warning,
                quality
            )
            self.output.setPlainText(report)
            self.colorize_output_status(quality.status)

        except ValueError as e:
            msg = "ERRORE DI INPUT\n\n" + str(e)
            self.output.setPlainText(msg)
            self.colorize_output_status("CRITICAL")
        except Exception as e:
            msg = "ERRORE DI CALCOLO\n\n" + str(e) + "\n\n" + traceback.format_exc()
            self.output.setPlainText(msg)
            self.colorize_output_status("CRITICAL")

    def build_quality_report(
        self,
        datum_mode: str,
        cylinder: CircleDatumInputResult,
        datum_c: Optional[CircleDatumInputResult],
        lateral_plane: Optional[PlaneFitResult],
        line_fit: Optional[LineFitResult],
        plane: PlaneFitResult,
        origin_real: np.ndarray,
        datum_real: np.ndarray,
        origin_nom: np.ndarray,
        datum_nom: np.ndarray,
        Zr: np.ndarray,
        Xr: np.ndarray
    ) -> QualityReport:
        thresholds = self.thresholds.values()
        lines = []
        severity = 0

        lines.append(f"Piano superiore: RMS={plane.rms:.6f}, Max={plane.max_residual:.6f}, AreaIndic={plane.area_indicator:.6f}")
        if plane.rms > thresholds["plane_rms_critical"]:
            lines.append("CRITICAL: errore RMS piano superiore oltre soglia critica.")
            severity = max(severity, 2)
        elif plane.rms > thresholds["plane_rms_warning"]:
            lines.append("WARNING: errore RMS piano superiore oltre soglia warning.")
            severity = max(severity, 1)

        if plane.area_indicator < thresholds["plane_area_warning"]:
            lines.append("WARNING: punti piano poco distribuiti, normale potenzialmente instabile.")
            severity = max(severity, 1)

        circle_features = [("Cilindro esterno", cylinder)]
        if datum_mode == "circle" and datum_c is not None:
            circle_features.append(("Datum C", datum_c))

        for label, feature in circle_features:
            if feature.source == "points" and feature.circle_fit is not None:
                cf = feature.circle_fit
                lines.append(
                    f"{label}: fit cerchio RMS={cf.rms:.6f}, Max={cf.max_residual:.6f}, "
                    f"Raggio={cf.radius:.6f}, N={cf.num_points}"
                )
                if cf.rms > thresholds["hole_rms_critical"]:
                    lines.append(f"CRITICAL: fit {label} oltre soglia critica.")
                    severity = max(severity, 2)
                elif cf.rms > thresholds["hole_rms_warning"]:
                    lines.append(f"WARNING: fit {label} oltre soglia warning.")
                    severity = max(severity, 1)
                if feature.imported_center_ignored:
                    lines.append(f"WARNING: {label} aveva anche un centro importato, ignorato perché sono presenti punti.")
                    severity = max(severity, 1)
            else:
                lines.append(f"{label}: centro inserito direttamente, nessun fit disponibile.")
                lines.append(f"WARNING: affidabilità di {label} dipende dal dato esterno.")
                severity = max(severity, 1)

        if datum_mode == "plane" and lateral_plane is not None:
            lines.append(
                f"Piano laterale Datum C: RMS={lateral_plane.rms:.6f}, "
                f"Max={lateral_plane.max_residual:.6f}, AreaIndic={lateral_plane.area_indicator:.6f}"
            )
            if lateral_plane.rms > thresholds["plane_rms_critical"]:
                lines.append("CRITICAL: errore RMS piano laterale oltre soglia critica.")
                severity = max(severity, 2)
            elif lateral_plane.rms > thresholds["plane_rms_warning"]:
                lines.append("WARNING: errore RMS piano laterale oltre soglia warning.")
                severity = max(severity, 1)

            normal_parallel_z = abs(float(np.dot(normalize(lateral_plane.normal), normalize(Zr))))
            lines.append(f"Piano laterale vs Z: |n laterale dot Z|={normal_parallel_z:.6f}")
            if normal_parallel_z > 0.95:
                lines.append("CRITICAL: piano laterale quasi parallelo al piano superiore, direzione X instabile.")
                severity = max(severity, 2)
            elif normal_parallel_z > 0.85:
                lines.append("WARNING: normale piano laterale poco inclinata rispetto a Z.")
                severity = max(severity, 1)

        if datum_mode == "line" and line_fit is not None:
            lines.append(
                f"Linea Datum C: RMS={line_fit.rms:.6f}, Max={line_fit.max_residual:.6f}, "
                f"N={line_fit.num_points}"
            )
            if line_fit.rms > thresholds["hole_rms_critical"]:
                lines.append("CRITICAL: fit linea Datum C oltre soglia critica.")
                severity = max(severity, 2)
            elif line_fit.rms > thresholds["hole_rms_warning"]:
                lines.append("WARNING: fit linea Datum C oltre soglia warning.")
                severity = max(severity, 1)

            line_parallel_z = abs(float(np.dot(normalize(line_fit.direction), normalize(Zr))))
            lines.append(f"Linea Datum C vs Z: |direzione linea dot Z|={line_parallel_z:.6f}")
            if line_parallel_z > 0.95:
                lines.append("CRITICAL: linea quasi parallela a Z, proiezione sul piano superiore instabile.")
                severity = max(severity, 2)
            elif line_parallel_z > 0.85:
                lines.append("WARNING: linea molto inclinata rispetto al piano superiore.")
                severity = max(severity, 1)

        if datum_mode == "circle":
            d_real = float(np.linalg.norm(datum_real - origin_real))
            d_nom = float(np.linalg.norm(datum_nom - origin_nom))
            diff_d = abs(d_real - d_nom)
            lines.append(f"Distanza centro cilindro -> Datum C nominale={d_nom:.6f}, reale={d_real:.6f}, delta={diff_d:.6f}")

            if d_real < thresholds["hole_distance_critical"]:
                lines.append("CRITICAL: Datum C troppo vicino al centro cilindro, frame instabile.")
                severity = max(severity, 2)

            if diff_d > thresholds["distance_delta_critical"]:
                lines.append("CRITICAL: differenza distanza nominale/reale oltre soglia critica.")
                severity = max(severity, 2)
            elif diff_d > thresholds["distance_delta_warning"]:
                lines.append("WARNING: differenza distanza nominale/reale oltre soglia warning.")
                severity = max(severity, 1)

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

    def build_report(
        self,
        datum_mode: str,
        cylinder: CircleDatumInputResult,
        datum_c: Optional[CircleDatumInputResult],
        lateral_plane: Optional[PlaneFitResult],
        line_fit: Optional[LineFitResult],
        plane: PlaneFitResult,
        plane_point_used: np.ndarray,
        plane_comp_vector: np.ndarray,
        plane_comp_diameter: float,
        plane_comp_radius: float,
        plane_comp_sign: int,
        origin_real: np.ndarray,
        datum_real: np.ndarray,
        origin_nom: np.ndarray,
        datum_nom: np.ndarray,
        real_frame: FrameResult,
        nominal_frame: FrameResult,
        R: np.ndarray,
        t: np.ndarray,
        T: np.ndarray,
        rotation_mode_label: str,
        rotation_lines: List[Tuple[str, float]],
        angle_output_warning: List[str],
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
        lines.append(f"ROTATE ({rotation_mode_label})")
        for axis_label, angle_value in rotation_lines:
            lines.append(f"{axis_label} = {angle_value:.6f}")
        if angle_output_warning:
            lines.append("")
            lines.extend(angle_output_warning)
        lines.append("")

        lines.append("REAL DATA")
        lines.append(f"Centro cilindro raw = {format_vec(cylinder.center_raw)}")
        lines.append(f"Origine reale projected = {format_vec(origin_real)}")
        if datum_mode == "circle" and datum_c is not None:
            lines.append(f"Centro Datum C raw = {format_vec(datum_c.center_raw)}")
            lines.append(f"Datum C projected = {format_vec(datum_real)}")
        elif datum_mode == "plane" and lateral_plane is not None:
            lines.append("Datum C mode = piano laterale")
            lines.append(f"Piano laterale point = {format_vec(lateral_plane.point)}")
            lines.append(f"Piano laterale normal raw = {format_vec(lateral_plane.normal)}")
        elif datum_mode == "line" and line_fit is not None:
            lines.append("Datum C mode = linea / asse")
            lines.append(f"Linea point = {format_vec(line_fit.point)}")
            lines.append(f"Linea direction raw = {format_vec(line_fit.direction)}")
        lines.append(f"Piano superiore point raw = {format_vec(plane.point)}")
        lines.append(f"Piano superiore point used = {format_vec(plane_point_used)}")
        lines.append(f"Piano superiore normal = {format_vec(real_frame.Z)}")
        if plane_comp_sign == 0 or plane_comp_diameter <= 0.0:
            lines.append("Compensazione piano = nessuna")
        else:
            comp_label = "+r lungo Z reale" if plane_comp_sign > 0 else "-r lungo Z reale"
            lines.append(
                f"Compensazione piano = {comp_label}, diametro sfera={plane_comp_diameter:.6f}, "
                f"raggio={plane_comp_radius:.6f}, vettore={format_vec(plane_comp_vector)}"
            )
        lines.append("")

        lines.append("CAD NOMINAL DATA")
        lines.append(f"Origine nominale = {format_vec(origin_nom)}")
        if datum_mode == "circle":
            lines.append(f"Datum C nominale = {format_vec(datum_nom)}")
        else:
            lines.append(f"Direzione Datum C nominale = {format_vec(nominal_frame.X)}")
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
        lines.append(f"Piano superiore RMS = {plane.rms:.6f}")
        lines.append(f"Piano superiore Max residual = {plane.max_residual:.6f}")
        lines.append(f"Piano superiore Area indicator = {plane.area_indicator:.6f}")

        circle_features = [("Cilindro esterno", cylinder)]
        if datum_mode == "circle" and datum_c is not None:
            circle_features.append(("Datum C", datum_c))

        for label, feature in circle_features:
            if feature.source == "points" and feature.circle_fit is not None:
                cf = feature.circle_fit
                lines.append(
                    f"{label}: fit da punti, center={format_vec(cf.center_3d)}, "
                    f"radius={cf.radius:.6f}, RMS={cf.rms:.6f}, Max={cf.max_residual:.6f}"
                )
            else:
                lines.append(f"{label}: centro inserito direttamente")

        if datum_mode == "plane" and lateral_plane is not None:
            lines.append(
                f"Piano laterale Datum C: normal={format_vec(lateral_plane.normal)}, "
                f"RMS={lateral_plane.rms:.6f}, Max={lateral_plane.max_residual:.6f}"
            )
        elif datum_mode == "line" and line_fit is not None:
            lines.append(
                f"Linea Datum C: direction={format_vec(line_fit.direction)}, "
                f"RMS={line_fit.rms:.6f}, Max={line_fit.max_residual:.6f}, N={line_fit.num_points}"
            )

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
# START WINDOW
# ============================================================

class StartWindow(QWidget):
    def __init__(self):
        super().__init__()
        window_icon = QIcon(str(asset_path("logo777.ico")))
        if not window_icon.isNull():
            self.setWindowIcon(window_icon)
        self.setWindowTitle("CenterTouch - Selezione datum")
        self.resize(1300, 900)

        self.stack = QStackedWidget()
        self.two_holes_tool = None
        self.cylinder_page = None

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.stack)

        self.stack.addWidget(self.build_home_page())

    def build_home_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        page_layout.addWidget(scroll_area)

        content = QWidget()
        scroll_area.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setSpacing(14)

        header_logo_size = 68
        header = QHBoxLayout()

        sire_block = QVBoxLayout()
        sire_logo = QLabel()
        sire_logo.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        sire_pixmap = QPixmap(str(asset_path("logo777_black on transparent.png")))
        if not sire_pixmap.isNull():
            sire_logo.setPixmap(
                sire_pixmap.scaled(header_logo_size, header_logo_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        sire_block.addWidget(sire_logo)

        title = QLabel("Centraggio parte in Space Meltio")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold;")

        wire_block = QVBoxLayout()
        wire_logo = QLabel()
        wire_logo.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        wire_pixmap = QPixmap(str(asset_path("Wire-trading.png")))
        if not wire_pixmap.isNull():
            wire_logo.setPixmap(
                wire_pixmap.scaled(header_logo_size, header_logo_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        wire_block.addWidget(wire_logo)

        header.addLayout(sire_block, 1)
        header.addWidget(title, 2)
        header.addLayout(wire_block, 1)
        layout.addLayout(header)

        datum_pixmap = QPixmap(str(asset_path("datum.png")))
        choices_row = QHBoxLayout()
        choices_row.setSpacing(18)
        choice_width = 300

        self.two_holes_btn = self.build_choice_button("Trasforma due fori")
        self.cylinder_btn = self.build_choice_button("Trasforma cilindro")
        self.on_demand_btn = self.build_choice_button("On demand")

        if datum_pixmap.isNull():
            missing_label = QLabel("Immagine datum.png non trovata.")
            missing_label.setAlignment(Qt.AlignCenter)
            missing_label.setStyleSheet("color: #900; font-weight: bold;")
            layout.addWidget(missing_label, 1)
        else:
            segment_width = datum_pixmap.width() // 3
            buttons = [self.two_holes_btn, self.cylinder_btn, self.on_demand_btn]
            for idx, button in enumerate(buttons):
                column = QVBoxLayout()
                column.setSpacing(8)

                x = idx * segment_width
                width = segment_width if idx < 2 else datum_pixmap.width() - x
                segment = datum_pixmap.copy(x, 0, width, datum_pixmap.height())

                scaled_segment = segment.scaledToWidth(choice_width, Qt.SmoothTransformation)

                image_label = QLabel()
                image_label.setAlignment(Qt.AlignCenter)
                image_label.setFixedSize(choice_width, scaled_segment.height())
                image_label.setPixmap(scaled_segment)

                button.setFixedWidth(choice_width)
                column.addWidget(image_label)
                column.addWidget(button)
                choices_row.addLayout(column)

            layout.addLayout(choices_row, 1)

        for button in (self.two_holes_btn, self.cylinder_btn, self.on_demand_btn):
            button.setMinimumHeight(30)
            button.setStyleSheet("font-size: 12px; font-weight: bold;")

        self.two_holes_btn.clicked.connect(self.open_two_holes_tool)
        self.cylinder_btn.clicked.connect(self.open_cylinder_placeholder)
        self.on_demand_btn.clicked.connect(self.show_on_demand_message)

        footer = QLabel("This software is licensed by SiRe, VAT No. IT01314390251, for use by Wire Trading.")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #666; font-size: 9px;")
        layout.addWidget(footer)

        return page

    def build_choice_button(self, text: str) -> QPushButton:
        button = QPushButton(text)
        button.setMinimumHeight(30)
        button.setStyleSheet("font-size: 12px; font-weight: bold;")
        return button

    def open_two_holes_tool(self):
        if self.two_holes_tool is None:
            self.two_holes_tool = MeltioFrameTool()
            self.stack.addWidget(self.two_holes_tool)
        self.stack.setCurrentWidget(self.two_holes_tool)

    def open_cylinder_placeholder(self):
        if self.cylinder_page is None:
            self.cylinder_page = CylinderFrameTool()
            self.stack.addWidget(self.cylinder_page)

        self.stack.setCurrentWidget(self.cylinder_page)

    def show_on_demand_message(self):
        QMessageBox.information(
            self,
            "Modulo on demand",
            "Modulo disponibile su richiesta.\n\n"
            "Questa configurazione datum richiede una procedura dedicata.\n"
            "Contattare Wire Trading e SiRe per valutare l'attivazione o lo sviluppo "
            "del modulo specifico per il componente."
        )


# ============================================================
# MAIN
# ============================================================

def main():
    app = QApplication(sys.argv)
    app_icon = QIcon(str(asset_path("logo777.ico")))
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    win = StartWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
