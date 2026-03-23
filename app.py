#!/usr/bin/env python
# coding: utf-8

import panel as pn
import numpy as np
from holoviews import opts
import holoviews as hv
from scipy.linalg import eigvals, eig
from scipy.integrate import solve_ivp
from bokeh.models import CrosshairTool

pn.extension(design='material')
pn.extension('katex', 'mathjax')


range_min = -10.0
range_max = 10.0
# Compute grid over a larger area so zooming out still shows vectors
grid_min = -30.0
grid_max = 30.0
step_size = 1.0
xs, ys = np.arange(grid_min, grid_max, step_size), np.arange(grid_min, grid_max, step_size)
X, Y = np.meshgrid(xs, ys)
def dx(a, b):
    return a*X + b*Y
def dy(c, d):
    return c*X + d*Y

def phase_portrait(a, b, c, d):
    U = dx(a, b)
    V = dy(c, d)
    return U, V


crosshair = CrosshairTool(dimensions="both")
def hook(plot, element):
    plot.state.add_tools(crosshair)


# --- Presets ---
PRESETS = {
    "(custom)": None,
    "Saddle": (2, 1, 1, -1),
    "Stable Node": (-2, 1, 0, -1),
    "Unstable Node": (2, 1, 0, 1),
    "Center": (0, 1, -1, 0),
    "Stable Spiral": (-1, 2, -2, -1),
    "Unstable Spiral": (1, 2, -2, 1),
    "Star Node (stable)": (-3, 0, 0, -3),
    "Star Node (unstable)": (3, 0, 0, 3),
}


# --- Classification ---
def classify_equilibrium(a, b, c, d):
    T = a + d
    D = a*d - b*c
    disc = T**2 - 4*D
    if D < 0:
        return "Saddle Point"
    if D == 0:
        return "Non-isolated / Degenerate"
    # D > 0
    if disc > 0:
        if T < 0:
            return "Stable Node"
        elif T > 0:
            return "Unstable Node"
        else:
            return "Stable Node / Center (borderline)"
    elif disc < 0:
        if T < 0:
            return "Stable Spiral"
        elif T > 0:
            return "Unstable Spiral"
        else:
            return "Center"
    else:
        # disc == 0
        if T < 0:
            return "Stable Star/Degenerate Node"
        elif T > 0:
            return "Unstable Star/Degenerate Node"
        else:
            return "Degenerate"


def d_system(t, state, a, b, c, d):
    x, y = state
    dx = a*x + b*y
    dy = c*x + d*y
    return [dx, dy]


def create_phase_plot(a, b, c, d, x, y, t):
    U, V = phase_portrait(a, b, c, d)
    mag = np.sqrt(U**2 + V**2)
    mag_safe = np.where(mag == 0, 1, mag)
    angle = (np.pi/2.) - np.arctan2(U/mag_safe, V/mag_safe)
    opts.defaults(opts.Scatter(color="red", size=10))

    resp_opts = dict(responsive=True, min_height=500)

    vectorfield = hv.VectorField((xs, ys, angle, mag)).opts(shared_axes=False, **resp_opts)
    # find any eigenvectors
    matrix = np.array([[a, b], [c, d]])
    _, eigs = eig(matrix)
    ds = hv.Dataset(np.linspace(grid_min, grid_max), 'x')

    # Trajectory overlay
    trajectory = None
    try:
        clip = grid_max
        p = (a, b, c, d)
        y0 = [x, y]
        t_span = (0.0, t)
        t_eval = np.arange(0.0, t, 0.01)
        if len(t_eval) > 0:
            result_ivp = solve_ivp(d_system, t_span, y0, args=p, method="LSODA", t_eval=t_eval)
            traj_x = np.clip(result_ivp.y[0], -clip, clip)
            traj_y = np.clip(result_ivp.y[1], -clip, clip)
            trajectory = hv.Curve(list(zip(traj_x, traj_y)), kdims=['x'], vdims=['y']).opts(
                color="red", line_width=2, shared_axes=False, **resp_opts)
            endpoint = hv.Scatter([(traj_x[-1], traj_y[-1])]).opts(
                color="red", size=8, marker="circle", shared_axes=False, **resp_opts)
            trajectory = trajectory * endpoint
    except Exception:
        pass

    # if the eigenvalues are not complex, find straight-line solutions
    if True in np.iscomplex(eigs):
        scatter = hv.Scatter([(x, y)]).opts(
            xlim=(range_min, range_max), ylim=(range_min, range_max),
            shared_axes=False, color="blue", **resp_opts)
        out = scatter * vectorfield.opts(shared_axes=False, hooks=[hook], **resp_opts)
        if trajectory:
            out = out * trajectory
        out.opts(shared_axes=False, **resp_opts, title="Phase Portrait DY/Dt=AY")
        return out
    else:
        slope_1 = eigs[1][0] / eigs[0][0]
        slope_2 = eigs[1][1] / eigs[0][1]

        expr1 = (hv.dim('x')*slope_1)
        expr2 = (hv.dim('x')*slope_2)

        sl_1 = ds.transform(y=expr1)
        sl_2 = ds.transform(y=expr2)

        sl_curve1 = hv.Curve(sl_1).opts(
            xlim=(range_min, range_max), ylim=(range_min, range_max),
            shared_axes=False, color="blue", **resp_opts)
        sl_curve2 = hv.Curve(sl_2).opts(
            xlim=(range_min, range_max), ylim=(range_min, range_max),
            shared_axes=False, color="blue", **resp_opts)
        scatter = hv.Scatter([(x, y)]).opts(
            xlim=(range_min, range_max), ylim=(range_min, range_max),
            shared_axes=False, color="blue", **resp_opts)
        out = vectorfield * sl_curve1 * sl_curve2 * scatter
        if trajectory:
            out = out * trajectory
        out.opts(shared_axes=False, hooks=[hook], title="Phase Portrait DY/Dt=AY", **resp_opts)
        return out


def plot_ivp_solns(a, b, c, d, t, x, y):
    p = (a, b, c, d)
    y0 = [x, y]
    t_span = (0.0, t)
    t_arr = np.arange(0.0, t, 0.01)
    if len(t_arr) == 0:
        return pn.pane.Markdown("Set time range > 0")
    result_ivp = solve_ivp(d_system, t_span, y0, args=p, method="LSODA", t_eval=t_arr)
    resp_opts = dict(responsive=True, min_height=350)
    out_x = hv.Curve([(t_arr[i], result_ivp.y[0, i]) for i in range(len(t_arr))]).opts(
        color="blue", xlabel="t", ylabel="x", title="x(t)", **resp_opts)
    out_y = hv.Curve([(t_arr[i], result_ivp.y[1, i]) for i in range(len(t_arr))]).opts(
        color="red", xlabel="t", ylabel="y", title="y(t)", **resp_opts)
    out_both = out_x * out_y
    out_both.opts(xlabel="t", ylabel="x,y", show_legend=True, title="x(t),y(t)", **resp_opts)
    return pn.Row(out_x + out_y + out_both, align="center", sizing_mode='stretch_width')


# --- T-D Plane with colored regions ---
_td_T = np.linspace(-10, 10, 200)
_td_parabola = _td_T**2 / 4
_td_D_max = 30  # upper bound for filled regions
_td_D_min = -12  # lower bound

def _build_td_regions():
    """Pre-build the static colored region polygons for the T-D plane."""
    regions = []

    # Saddle: entire strip where D < 0
    saddle = hv.Polygons([{
        'x': [-10, 10, 10, -10],
        'y': [0, 0, _td_D_min, _td_D_min],
    }]).opts(color='#f4c542', alpha=0.25, line_width=0)

    # Stable Node: T < 0, 0 < D < T^2/4
    t_left = _td_T[_td_T <= 0]
    par_left = t_left**2 / 4
    stable_node = hv.Polygons([{
        'x': list(t_left) + list(t_left[::-1]),
        'y': list(par_left) + [0]*len(t_left),
    }]).opts(color='#2196F3', alpha=0.20, line_width=0)

    # Unstable Node: T > 0, 0 < D < T^2/4
    t_right = _td_T[_td_T >= 0]
    par_right = t_right**2 / 4
    unstable_node = hv.Polygons([{
        'x': list(t_right) + list(t_right[::-1]),
        'y': list(par_right) + [0]*len(t_right),
    }]).opts(color='#f44336', alpha=0.20, line_width=0)

    # Stable Spiral: T < 0, D > T^2/4
    stable_spiral = hv.Polygons([{
        'x': list(t_left) + list(t_left[::-1]),
        'y': [_td_D_max]*len(t_left) + list(par_left[::-1]),
    }]).opts(color='#4CAF50', alpha=0.20, line_width=0)

    # Unstable Spiral: T > 0, D > T^2/4
    unstable_spiral = hv.Polygons([{
        'x': list(t_right) + list(t_right[::-1]),
        'y': [_td_D_max]*len(t_right) + list(par_right[::-1]),
    }]).opts(color='#FF9800', alpha=0.20, line_width=0)

    # Center line: T=0, D>0 — draw as a vertical line
    center_line = hv.Curve([(0, 0), (0, _td_D_max)]).opts(
        color='purple', line_dash='dashed', line_width=2)

    # D=0 horizontal line
    d_zero_line = hv.Curve([(-10, 0), (10, 0)]).opts(
        color='gray', line_dash='dotted', line_width=1)

    # Parabola D = T^2/4
    parabola = hv.Curve(list(zip(_td_T, _td_parabola))).opts(
        color='black', line_width=2)

    # Labels
    labels = hv.Overlay([
        hv.Text(0, -6, "Saddle").opts(text_color='#b8960f', text_font_size="11pt", text_font_style='bold'),
        hv.Text(-6.5, 8, "Stable\nNode").opts(text_color='#1565C0', text_font_size="10pt", text_font_style='bold'),
        hv.Text(6.5, 8, "Unstable\nNode").opts(text_color='#c62828', text_font_size="10pt", text_font_style='bold'),
        hv.Text(-5, 22, "Stable\nSpiral").opts(text_color='#2E7D32', text_font_size="10pt", text_font_style='bold'),
        hv.Text(5, 22, "Unstable\nSpiral").opts(text_color='#E65100', text_font_size="10pt", text_font_style='bold'),
        hv.Text(0.8, 27, "Centers").opts(text_color='purple', text_font_size="9pt", text_font_style='italic'),
    ])

    return (saddle * stable_node * unstable_node * stable_spiral * unstable_spiral
            * d_zero_line * parabola * center_line * labels)

_td_static = _build_td_regions()

def trace(a, b, c, d):
    return a + d

def determinant(a, b, c, d):
    return a*d - b*c

def td_point(a, b, c, d):
    resp_opts = dict(responsive=True, min_height=500)
    point = hv.Scatter([(trace(a, b, c, d), determinant(a, b, c, d))]).opts(
        shared_axes=False, color="black", size=12, marker="circle", line_color="white",
        line_width=2, **resp_opts)

    out = _td_static * point
    out.opts(shared_axes=False, ylabel="D", xlabel="T", title="Trace-Determinant Plane",
             xlim=(-10, 10), ylim=(_td_D_min, _td_D_max), **resp_opts)
    return out


def eigenvalues(a, b, c, d):
    matrix = np.array([[a, b], [c, d]])
    eigs = eigvals(matrix)
    if (eigs[0].imag != 0j) or (eigs[1].imag != 0j):
        return pn.Column(
            pn.pane.LaTeX(r"$\lambda_1={:.3f}$".format(eigs[1])),
            pn.pane.LaTeX(r"$\lambda_2={:.3f}$".format(eigs[0])))
    else:
        return pn.Column(
            pn.pane.LaTeX(r"$\lambda_1={:.3f}$".format(eigs[1].real)),
            pn.pane.LaTeX(r"$\lambda_2={:.3f}$".format(eigs[0].real)))


def matrix_display(a, b, c, d):
    return pn.pane.LaTeX(
        r"$A = \begin{{bmatrix}} {:.1f} & {:.1f} \\ {:.1f} & {:.1f} \end{{bmatrix}}$".format(a, b, c, d),
        renderer="katex", styles={"font-size": "18px"})


def classification_display(a, b, c, d):
    label = classify_equilibrium(a, b, c, d)
    return pn.pane.Markdown(
        f"### Equilibrium: **{label}**",
        styles={"color": "#333", "text-align": "center"})


# --- Widgets ---
a_widget = pn.widgets.EditableFloatSlider(name="a", value=1, start=range_min, end=range_max, step=0.1)
b_widget = pn.widgets.EditableFloatSlider(name="b", value=1, start=range_min, end=range_max, step=0.1)
c_widget = pn.widgets.EditableFloatSlider(name="c", value=1, start=range_min, end=range_max, step=0.1)
d_widget = pn.widgets.EditableFloatSlider(name="d", value=1, start=range_min, end=range_max, step=0.1)
x_widget = pn.widgets.EditableFloatSlider(name="x0", value=1, start=range_min, end=range_max, step=0.1)
y_widget = pn.widgets.EditableFloatSlider(name="y0", value=1, start=range_min, end=range_max, step=0.1)
t_widget = pn.widgets.EditableFloatSlider(name="time range", value=6, start=0, end=20, step=0.1)

# --- Preset dropdown ---
preset_widget = pn.widgets.Select(name="Preset Examples", options=list(PRESETS.keys()), value="(custom)")

def on_preset_change(event):
    vals = PRESETS.get(event.new)
    if vals is None:
        return
    a_widget.value, b_widget.value, c_widget.value, d_widget.value = vals

preset_widget.param.watch(on_preset_change, 'value')


# --- Bindings ---
bound_plot = pn.bind(create_phase_plot, a=a_widget, b=b_widget, c=c_widget, d=d_widget,
                     x=x_widget, y=y_widget, t=t_widget)
bound_solns = pn.bind(plot_ivp_solns, a=a_widget, b=b_widget, c=c_widget, d=d_widget,
                      t=t_widget, x=x_widget, y=y_widget)
td_plot = pn.bind(td_point, a=a_widget, b=b_widget, c=c_widget, d=d_widget)
eigen_indicator = pn.bind(eigenvalues, a=a_widget, b=b_widget, c=c_widget, d=d_widget)
matrix_pane = pn.bind(matrix_display, a=a_widget, b=b_widget, c=c_widget, d=d_widget)
classification_pane = pn.bind(classification_display, a=a_widget, b=b_widget, c=c_widget, d=d_widget)


# --- Layout ---
test_app = pn.Column(
    pn.pane.Markdown("# Phase Portraits for Linear ODE Systems", align="center"),
    pn.Row(
        pn.Column(
            pn.pane.Markdown("## Constant Matrix A"),
            pn.pane.LaTeX(r"$\frac{dY}{dt}=AY$", renderer="katex"),
            preset_widget,
            a_widget, b_widget, c_widget, d_widget,
            sizing_mode='stretch_width',
        ),
        pn.Column(
            matrix_pane,
            pn.pane.Markdown("---"),
            pn.pane.Markdown("### Eigenvalues"),
            eigen_indicator,
            pn.pane.Markdown("---"),
            classification_pane,
            align="center",
            sizing_mode='stretch_width',
        ),
        sizing_mode='stretch_width',
    ),
    pn.Row(bound_plot, td_plot, sizing_mode='stretch_width', min_height=500),
    pn.pane.Markdown("## Initial Value Problem", align="center"),
    pn.Row(x_widget, y_widget, t_widget, align="center", sizing_mode='stretch_width'),
    bound_solns,
    sizing_mode='stretch_width',
)

test_app.servable()
