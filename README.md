# Manifold-SOS

Implementation of the Riemannian Gradient Descent for the problem (1.2) of my Master's thesis, corresponding to a linear sum-of-squares program.
For this, we rely on the Manopt.jl package.

To launch the program (you need to have Julia installed):
```
julia --project manifold_SOS.jl
```

The problem considered as example is

$$\min_{u_1, u_2 \in \mathbb{R}[x]_1} \frac{d}{dx} (u_1^2(x) + u_2^2(x)) \mid_{x = 0}$$

such that

$$\frac{d^2}{dx^2} (u_1^2(x) + u_2^2(x)) \mid_{x = 0} = a$$

and

$$u_1^2(0) + u_2^2(0) = c$$

are fixed ($a > 0$).

The optimal solution to this problem is $-2 \sqrt{ac}$
