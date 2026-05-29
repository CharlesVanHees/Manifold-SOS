"""
Master's thesis Charles Van Hees
Sum-of-squares programming via optimization on manifolds

Riemannian Gradient Descent algorithm for problem (1.2) on the pairs of coprime polynomials of degree d
"""

using LinearAlgebra
using ManifoldsBase, Manopt
using ManifoldsBase: ℝ

"""
    SumOfSquaresMan <: AbstractManifold{ℝ}

    Define a set of polynomials whose sum of squares is equal to a given polynomial.
"""
struct SumOfSquaresMan{T <: Real} <: AbstractManifold{ℝ}
    degree::Int64 # Degree of the two polynomials
    L::Matrix{T}  # Linear operator defining the set of constraints
    b::Vector{T}  # Constraint vector

    function SumOfSquaresMan(degree::Int64, L::Matrix{T}, b::Vector{T}) where {T <: Real}
        size(L) == (length(b), 2*degree+1) || throw(ArgumentError("Size of the operator L $(size(L)) does not match with the size of vector b ($(length(b))) and a 2d-degree polynomial, where d = $(degree)"))
        return new{T}(degree, L, b)
    end
end

ManifoldsBase.representation_size(M::SumOfSquaresMan) = (2*(M.degree+1),)

function ManifoldsBase.manifold_dimension(M::SumOfSquaresMan)
    m = size(b) # We assume L is surjective
    return 2*M.degree + 1 - m + 1 # The +1 follows from the coprimality of the polynomials
end

"""
    Return the matrix associated to the operator A_u, expressed in the monomial basis.
    This corresponds to the Sylvester matrix slightly modified

    u = [u_1; u_2] is a stacked vector containing the coefficients of both polynomials.
"""
function A_u(u::AbstractVector)
    d = div(length(u), 2) - 1
    A_u = zeros(eltype(u), 2d + 1, 2d + 2)
    for i in 0:1
        for j in 0:d
            for k in 0:d
                A_u[j + k + 1, i * (d + 1) + j + 1] = u[i * (d + 1) + k + 1]
            end
        end
    end
    return A_u
end
σ(u::AbstractVector) = A_u(u) * u

"""
    Check whether a point is in the manifold
"""
function ManifoldsBase.check_point(M::SumOfSquaresMan, u; kwargs...)
    if !isapprox(M.L * σ(u), M.b; atol=1e-10, kwargs...)
        return DomainError(M.L * σ(u), "L(σ(u)) is $(M.L * σ(u)), which is not b = $(M.b).")
    end
end

"""
    Check whether a vector is in the tangent space at u
"""
function ManifoldsBase.check_vector(M::SumOfSquaresMan, u, v; kwargs...)
    if !isapprox(2 * M.L * A_u(u) * v, zeros(length(M.b)); atol=1e-10, kwargs...)
        return DomainError(2 * M.L * A_u(u) * v, "v = $(v) is not a tangent vector to the manifold M at point u = $(u).")
    end
end

"""
    Returns an objective function and its Riemannian gradient
"""
function objective(L::Vector{T}, degree::Int) where {T <: Real}
    length(L) == 2*degree + 1 || throw(ArgumentError("The objective function is not adapted for polynomials of degree $(2*degree + 1)."))
    
    function obj(M::SumOfSquaresMan, u::Vector{T}) where {T <: Real}
        ManifoldsBase.is_point(M, u, error = :error)
        return L' * A_u(u) * u
    end

    function proj_grad(M::SumOfSquaresMan, u::Vector{T}) where {T <: Real}
        ManifoldsBase.is_point(M, u, error = :error)
        A_uu = A_u(u)
        grad = 2 * A_uu' * L # L is represented as a column vector, even if it is a row vector. Hence, when mathematically we consider the transpose, we keep L as it is a column vector.
        Dh = 2 * M.L * A_uu
        return grad - Dh' * (Dh' \ grad)
    end

    return obj, proj_grad
end

"""
    Newton-like retraction
"""
struct NewtonRetraction <: ManifoldsBase.ApproximateRetraction 
    max_iter::Int64
    atol::Float64
end
function ManifoldsBase.retract!(M::SumOfSquaresMan, q::Vector{Float64}, u::Vector{Float64}, v::Vector{Float64}, method::NewtonRetraction)
    q .= u + v
    for _ in 1:method.max_iter
        q .-= 2*M.L*A_u(q) \ (M.L*σ(q) - M.b)
        if norm(M.L * σ(q) - M.b) <= method.atol break end
    end
    return q
end
function ManifoldsBase._retract_fused!(M::SumOfSquaresMan, q::Vector{Float64}, u::Vector{Float64}, v::Vector{Float64}, t::Float64, method::NewtonRetraction)
    return ManifoldsBase.retract!(M, q, u, t*v, method)
end

###################################################################
###################################################################
### Example
###################################################################

# We would like to minimize  b
#                  such that ax^2 + bx + c ≥ 0 for all x (i.e., it is a sum of squares)
#                            a, c fixed, with a > 0
# i.e., the discriminant b^2 - 4ac should be nonpositive.
# The minimum is obtained at b = -2sqrt(ac).

a,c = 17,93

M = SumOfSquaresMan(1, [1 0 0; 0 0 1], [a, c])
f, grad_f = objective([0, 1, 0], 1)
q0 = retract(M, rand(4), rand(4), NewtonRetraction(10000, 1e-12)) # find a feasible starting point
q1 = gradient_descent(M, f, grad_f, q0;
    retraction_method = NewtonRetraction(10000, 1e-12),
    stepsize = DecreasingLength(M; length=1.0),
    stopping_criterion = StopAfterIteration(50),
    X = zeros(4),
)

@show is_point(M, q1)
@show f(M,q1)
@show f(M,q1) ≈ -2*sqrt(a*c)