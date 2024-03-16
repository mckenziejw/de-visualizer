#!/usr/bin/env python
# coding: utf-8

# In[106]:


import panel as pn
import hvplot.pandas
import pandas as pd
import numpy as np
from holoviews import opts
import pathlib
import holoviews as hv
from scipy.linalg import eigvals, eig

pn.extension(design='material')
pn.extension('katex', 'mathjax')


# In[107]:


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


# In[108]:


def create_phase_plot(a, b, c, d):
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
        return vectorfield
    else:
        slope_1 = eigs[1][0] / eigs[0][0]
        slope_2 = eigs[1][1] / eigs[0][1]

        expr1 = (hv.dim('x')*slope_1)
        expr2 = (hv.dim('x')*slope_2)

        sl_1 = ds.transform(y=expr1)
        sl_2 = ds.transform(y=expr2)

        sl_curve1 = hv.Curve(sl_1).opts(xlim=(range_min, range_max), ylim=(range_min, range_max), shared_axes=False, color="blue", height=800, width=800)
        sl_curve2 = hv.Curve(sl_2).opts(xlim=(range_min, range_max), ylim=(range_min, range_max), shared_axes=False, color="blue", height=800, width=800)
        out = vectorfield*sl_curve1*sl_curve2
        out.opts(shared_axes = False, height=800, width=800)
        #vectorfield.relabel(label)
        return out


# In[110]:


a_widget = pn.widgets.FloatSlider(name="a", value = 1, start = range_min, end=range_max)
b_widget = pn.widgets.FloatSlider(name="b", value = 1, start = range_min, end=range_max)
c_widget = pn.widgets.FloatSlider(name="c", value = 1, start = range_min, end=range_max)
d_widget = pn.widgets.FloatSlider(name="d", value = 1, start = range_min, end=range_max)


# In[111]:


bound_plot = pn.bind(create_phase_plot, a=a_widget, b=b_widget, c=c_widget, d=d_widget)


# In[112]:


ds = hv.Dataset(np.linspace(-10, 10), 't')
expr = (hv.dim('t')**2)/4
transformed = ds.transform(y=expr)

def trace(a, b, c, d):
    return a + d

def determinant(a, b, c, d):
    return a*d-b*c

def td_point(a, b, c, d):   
    first = hv.Curve(transformed).opts(shared_axes=False, height=800, width=800)
    second =  hv.Scatter([(trace(a, b, c, d), determinant(a, b, c, d))]).opts(shared_axes=False, height=800, width=800)
    out = first*second
    out.opts(shared_axes=False, height=800, width=800)
    return first*second



def eigenvalues(a,b,c,d):
    matrix = np.array([[a, b], [c,d]])
    eigs = eigvals(matrix)
    if (eigs[0].imag != 0j) or (eigs[1].imag !=0j):
        return pn.Column(pn.pane.LaTeX(r"$\lambda_1={:.3f}$".format(eigs[1])), pn.pane.LaTeX(r"$\lambda_2={:.3f}$".format(eigs[0])))
    else:
        return pn.Column(pn.pane.LaTeX(r"$\lambda_1={:.3f}$".format(eigs[1].real)), pn.pane.LaTeX(r"$\lambda_2={:.3f}$".format(eigs[0].real)))
    


# In[113]:


td_plot = pn.bind(td_point, a=a_widget, b=b_widget, c=c_widget, d=d_widget)


# In[114]:


eigen_indicator = pn.bind(eigenvalues, a=a_widget, b=b_widget, c=c_widget, d=d_widget)

test_app = pn.Column(
    pn.Row(pn.Column(a_widget, b_widget, c_widget, d_widget), eigen_indicator),
    pn.Row(bound_plot, td_plot)
)
# In[116]:


test_app.servable()

# In[ ]:




