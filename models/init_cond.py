import jax.numpy as jnp

def get_initial_condition(x, E, name):
    """_Initial condition specifier_

    Args:
        x (_type_): _mesh x array_
        E (_type_): _number of ensemble members to initialise_
        name (_type_): _name of initial condition_

    Returns:
        _type_: _array of shape (E,nx)
        consisting of E coppies of the initial condition specified
    """
    if name == 'sin':
        ans = jnp.sin(2 * jnp.pi * x)

    elif name == 'compact_bump':
        x_min = 0.2; x_max = 0.3
        s = (2 * (x - x_min) / (x_max - x_min)) - 1
        ans = jnp.exp(-1 / (1 - s**2)) * (jnp.abs(s) < 1)

    elif name == 'Kassam_Trefethen_KS_IC':
        ans = jnp.cos( x / 16 ) * ( 1 + jnp.sin(x / 16) )

    elif name == 'Kassam_Trefethen_KdV_IC_eq3pt1':
        A = 25
        B = 16
        ans  = 3 * A**2 * jnp.cosh( 0.5 * A * (x+2) )**-2
        ans += 3 * B**2 * jnp.cosh( 0.5 * B * (x+1) )**-2

    elif name == 'traveling_wave':
        beta = 3
        ans  =  12 * beta**2 * jnp.cosh( beta * ( x ) )**-2
        ans =  jnp.where(ans<1e-7,0,ans)#
    elif name == 'gaussian':
        A = 1; x0 = 0.5; sigma = 0.1
        ans = A * jnp.exp(-((x - x0)**2) / (2 * sigma**2))

    else:
        raise ValueError(f"Initial condition {name} not recognised")
    
    ic = jnp.tile(ans, (E, 1))
    return ic