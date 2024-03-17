#!/usr/bin/env python
# coding: utf-8

# In[43]:


import panel as pn
import hvplot.pandas
import pandas as pd
import numpy as np
from holoviews import opts, streams
import pathlib
import holoviews as hv
from scipy.linalg import eigvals, eig
from scipy.integrate import solve_ivp
from bokeh.models import CrosshairTool

pn.extension(design='material')
pn.extension('katex', 'mathjax')


# In[44]:


range_min = -10.0
range_max = 10.0
step_size = 0.5
xs, ys = np.arange(range_min, range_max, step_size), np.arange(range_min, range_max, step_size)
X, Y = np.meshgrid(xs, ys)
def dx(a, b):
    return a*X + b*Y
def dy(c, d):
    return c*X + d*Y

def phase_portrait(a, b, c, d):
    U = dx(a, b)
    V = dy(c, d)
    return U, V


# In[45]:


crosshair = CrosshairTool(dimensions="both")
def hook(plot, element):
    plot.state.add_tools(crosshair)
    


# In[46]:


def create_phase_plot(a, b, c, d, x, y):
    U,V = phase_portrait(a, b, c, d)
    mag = np.sqrt(U**2 + V**2)
    angle = (np.pi/2.) - np.arctan2(U/mag, V/mag)
    opts.defaults(opts.Scatter(color="red",size=10))

    vectorfield = hv.VectorField((xs, ys, angle, mag)).opts(shared_axes=False, height=800, width=800)
    # find any eigenvectors
    matrix = np.array([[a, b], [c,d]])
    _, eigs = eig(matrix)
    ds = hv.Dataset(np.linspace(range_min, range_max), 'x')
    
    # if the eigenvalues are not complex, find straight-line solutions
    if True in np.iscomplex(eigs):
        scatter = hv.Scatter([(x,y)]).opts(xlim=(range_min, range_max), ylim=(range_min, range_max), shared_axes=False, color="blue", height=800, width=800)
        return scatter*vectorfield.opts(shared_axes=False, height=800, width=800, hooks=[hook])
    else:
        slope_1 = eigs[1][0] / eigs[0][0]
        slope_2 = eigs[1][1] / eigs[0][1]

        expr1 = (hv.dim('x')*slope_1)
        expr2 = (hv.dim('x')*slope_2)

        sl_1 = ds.transform(y=expr1)
        sl_2 = ds.transform(y=expr2)

        sl_curve1 = hv.Curve(sl_1).opts(xlim=(range_min, range_max), ylim=(range_min, range_max), shared_axes=False, color="blue", height=800, width=800)
        sl_curve2 = hv.Curve(sl_2).opts(xlim=(range_min, range_max), ylim=(range_min, range_max), shared_axes=False, color="blue", height=800, width=600)
        scatter = hv.Scatter([(x,y)]).opts(xlim=(range_min, range_max), ylim=(range_min, range_max), shared_axes=False, color="blue", height=800, width=800)
        out = vectorfield*sl_curve1*sl_curve2*scatter
        out.opts(shared_axes = False, height=800, width=800, hooks=[hook], title="Phase Portrait DY/Dt=AY")
        
        #vectorfield.relabel(label)
        return out


# In[88]:


def d_system(t, state, a, b, c, d):
    x, y = state
    dx = a*x + b*y
    dy = c*x + d*y
    return [dx, dy]

def plot_ivp_solns(a, b, c, d, t, x, y):
    p = (a, b, c, d)
    y0 = [x, y]
    t_span = (0.0, t)
    t = np.arange(0.0, t, 0.01)
    result_ivp = solve_ivp(d_system, t_span, y0, args=p, method="LSODA", t_eval=t)
    out_x = hv.Curve((t[i], result_ivp.y[0, i]) for i in range(len(t))).opts(color="blue", xlabel="t", ylabel="x", height=500, width=500, title="x(t)")
    out_y = hv.Curve((t[i], result_ivp.y[1, i]) for i in range(len(t))).opts(color="red", xlabel="t", ylabel="y", height=500, width=500, title="y(t)")
    out_both = out_x*out_y
    out_both.opts(xlabel="t", ylabel="x,y", show_legend=True, height=500, width=500, title="x(t),y(t)")
    return pn.Row(out_x+out_y+out_both, align="center")


# In[89]:


a_widget = pn.widgets.FloatSlider(name="a", value = 1, start = range_min, end=range_max)
b_widget = pn.widgets.FloatSlider(name="b", value = 1, start = range_min, end=range_max)
c_widget = pn.widgets.FloatSlider(name="c", value = 1, start = range_min, end=range_max)
d_widget = pn.widgets.FloatSlider(name="d", value = 1, start = range_min, end=range_max)
x_widget = pn.widgets.FloatSlider(name="x0", value = 1, start = range_min, end=range_max) 
y_widget = pn.widgets.FloatSlider(name="y0", value = 1, start = range_min, end=range_max) 
t_widget = pn.widgets.FloatSlider(name="time range", value = 6, start = 0, end=20)


# In[90]:


bound_plot = pn.bind(create_phase_plot, a=a_widget, b=b_widget, c=c_widget, d=d_widget, x=x_widget, y=y_widget)


# In[91]:


bound_solns = pn.bind(plot_ivp_solns,  a=a_widget, b=b_widget, c=c_widget, d=d_widget, t=t_widget, x=x_widget, y=y_widget)


# In[92]:


ds = hv.Dataset(np.linspace(-10, 10), 't')
expr = (hv.dim('t')**2)/4
transformed = ds.transform(y=expr)

def trace(a, b, c, d):
    return a + d

def determinant(a, b, c, d):
    return a*d-b*c

def td_point(a, b, c, d):   
    first = hv.Curve(transformed).opts(shared_axes=False)
    second =  hv.Scatter([(trace(a, b, c, d), determinant(a, b, c, d))]).opts(shared_axes=False)
    out = first*second
    out.opts(shared_axes=False, height=800, width=800, ylabel="D", xlabel="T", title="Trace Determinant Plane")
    return out



def eigenvalues(a,b,c,d):
    matrix = np.array([[a, b], [c,d]])
    eigs = eigvals(matrix)
    if (eigs[0].imag != 0j) or (eigs[1].imag !=0j):
        return pn.Column(pn.pane.LaTeX(r"$\lambda_1={:.3f}$".format(eigs[1])), pn.pane.LaTeX(r"$\lambda_2={:.3f}$".format(eigs[0])))
    else:
        return pn.Column(pn.pane.LaTeX(r"$\lambda_1={:.3f}$".format(eigs[1].real)), pn.pane.LaTeX(r"$\lambda_2={:.3f}$".format(eigs[0].real)))
    


# In[93]:


td_plot = pn.bind(td_point, a=a_widget, b=b_widget, c=c_widget, d=d_widget)


# In[94]:


eigen_indicator = pn.bind(eigenvalues, a=a_widget, b=b_widget, c=c_widget, d=d_widget)


# In[95]:


test_app = pn.Column(pn.pane.Markdown("# Phase portraits for Linear ODE systems", align="center"),
    pn.Row(pn.Column(pn.pane.Markdown("## Set values for constant matrix A of form"),pn.pane.LaTeX(r"$\frac{dY}{dt}=AY\:\:\:\:\:\:\:\:A=\begin{bmatrix}a & b\\c&d\end{bmatrix}$", renderer="katex"),a_widget, b_widget, c_widget, d_widget), pn.Column(pn.pane.Markdown("## Calculated eigenvalues:"),eigen_indicator, align="center"), align="center"),
    pn.Row(bound_plot, td_plot, align="center"),
    pn.Row(pn.pane.Markdown("## Set Initial Value parameters: ", align="center")),
    pn.Row(x_widget, y_widget, t_widget, align="center"),
    bound_solns)


# In[96]:


test_app.servable()


# In[ ]:




