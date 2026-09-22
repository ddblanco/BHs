"""Euler variations of the generic Einstein action, before fixing g=r².

Conventions: E_v=(dL/dv-d/dr(dL/dv'))/sqrt(b*h/f).
The f variation is reserved as a constraint. No reference solution is used.
"""
import numpy as np


def radial_equations(r, values, first, second):
    """Return E_b,E_f,E_g,E_h,E_w for radial jets ordered b,f,h,w.

    Valid in the regular exterior (b,f,h>0). Arrays broadcast over points.
    f'' and w itself do not enter the equations.
    """
    b,f,h,w = np.asarray(values)
    bp,fp,hp,wp = np.asarray(first)
    bpp,_,hpp,wpp = np.asarray(second)
    eb = -(2*r**4*b*f*h*hpp-r**4*b*f*hp**2+r**4*b*h*fp*hp
           +r**4*f*h**3*wp**2+4*r**3*b*f*h*hp+4*r**3*b*h**2*fp
           +4*r**2*b*f*h**2-16*r**2*b*h**2+4*b*h**3)/(4*r**2*b**2*h**2)
    ef = (r**4*f*h**2*wp**2+r**4*f*bp*hp+4*r**3*b*f*hp
          +4*r**3*f*h*bp+4*r**2*b*f*h-16*r**2*b*h+4*b*h**2)/(4*r**2*b*f*h)
    eg = -(2*r**4*b**2*f*h*hpp-r**4*b**2*f*hp**2+r**4*b**2*h*fp*hp
           -r**4*b*f*h**3*wp**2+2*r**4*b*f*h**2*bpp+r**4*b*f*h*bp*hp
           +r**4*b*h**2*bp*fp-r**4*f*h**2*bp**2+2*r**3*b**2*f*h*hp
           +2*r**3*b**2*h**2*fp+2*r**3*b*f*h**2*bp-4*b**2*h**3)/(2*r**4*b**2*h**2)
    eh = -(-3*r**4*b*f*h*wp**2+2*r**4*b*f*bpp+r**4*b*bp*fp
           -r**4*f*bp**2+4*r**3*b**2*fp+4*r**3*b*f*bp
           +4*r**2*b**2*f-16*r**2*b**2+12*b**2*h)/(4*r**2*b**2*h)
    ew = r*(-2*r*b*f*h*wpp-3*r*b*f*hp*wp-r*b*h*fp*wp
            +r*f*h*bp*wp-4*b*f*h*wp)/(2*b**2)
    return dict(b=eb,f=ef,g=eg,h=eh,w=ew)
