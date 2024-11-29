import jax.numpy as jnp

def Forward_model(q,A,dt,B,dW):
    """_Solve d_t q + A q dt + B dW = 0
    via Euler Maruyama_

    Args:
        q: _(E,nx)_
        A: _(nx,nx)_
        dt: _float_
        B: _(P,nx)_
        dW: _(P,)_

    Returns:
        _type_: _ _
    """
    B = jnp.einsum('ij,i->j',B,dW) 
    qnew = q + dt*jnp.einsum('ij,j->i',A,q) + B
    return qnew