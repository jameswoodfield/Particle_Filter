# Particle Filter

<img src="Saving/Cropped_KS.png" alt="drawing" width="200"/>
<img src="Saving/Cropped_KdV.png" alt="drawing" width="200"/>

### Example 1: KS EQUATION
In this .ipybn we run the Deterministic Kuramoto-Sivashinsky equation under the initial conditions specified in Kassam and Trefethen. 

### Example 2: 
In these two notebooks: We run deterministic Kuramoto-Sivashinsky and deterministic KdV equations under initial condition pertubations, and observe initial condition sensitivities of magnitudes $10^{8}$ and $10^{1}$. 

### Example 3: 
In these two notebooks: We demonstrate how to run a particle filter with the stochastic KS and KdV equation under transport noise. We include subsampled data in both space and time. 

These notebooks indicate a qualitive difference between the ability of the particle filter to converge between the KdV equation and the KS equation. In the context of the twin experiment. 

This is interesting in light of the Low dimensional behaviour of the KS equation, and the fact that the PF filter typically fails due to dimensionality of the state space being a challenge. The results in this example indicating that the equations sensitivity to initial condition and stochastic pertubation makes the KS more challenging. 


### Example 5: 
These two notebooks contain the temporal and spatial convergence of a pathwise stochastic travelling wave under constant noise. To demonstrate spectral convergence is difficult due to the spatial accuracy being near machine precision at moderate resolution, we have lowered the timestep drastically, but increased the spatial step. The temporal convergence is computed strongly, using a stochastic travelling wave. 

### Example 6: 
These notebooks simply visualise the deterministic solution, for the KdV and KS equation. 