# problem data
struct PROBLEM
    n::Int64
    m::Int64
    cl::Vector{Float64}
    cu::Vector{Float64}
    xl::Vector{Float64}
    xu::Vector{Float64}
    ceq::Union{UnitRange{Int64},Vector{Int64}}
    cineq::Union{UnitRange{Int64},Vector{Int64}}
    cineqLU::Union{UnitRange{Int64},Vector{Int64}}
    mLU::Int64
    ncc::Int64
    cc::Union{UnitRange{Int64},Vector{Int64}}
    cvar::Vector{Int64}
    # scaling factors
    wf::Float64
    wc::Vector{Float64}
end

# iterate
mutable struct WORK
    lam::Vector{Float64}
    rho::Float64
    x::Vector{Float64}
    f::Float64
    c::Vector{Float64}
    uns_c::Vector{Float64}
    gL::Vector{Float64}
    pgL::Vector{Float64}
    nwork::Vector{Float64}
    mwork::Vector{Float64}
    Lwork::Vector{Float64}
    # SPG/NPG
    xnew::Vector{Float64}
    gLnew::Vector{Float64}
    s::Vector{Float64}
    y::Vector{Float64}
    xbest::Vector{Float64}
    lastL::Vector{Float64}
    spectral::Float64
end

# output information
struct INFO
    status::Int64
    iter::Int64
    sol::Vector{Float64}
    f::Float64
    infeas::Float64
    gL_supnorm::Float64
    infeas_compl::Float64
    lam::Vector{Float64}
    max_lam_supnorm::Float64
    final_rho::Float64
    time::Float64
end

# parameters not present in the header of the function 'al'
mutable struct EXTRA_PAR
    # AL extra parameters:
    tau::Float64
    theta::Float64
    lammin::Float64
    lammax::Float64
    rhomax::Float64
    rhoinimin::Float64
    rhoinimax::Float64
    fmin::Float64
    # SPG/NPG parameters:
    eta::Float64
    lsm::Int64
    spg_maxit::Int64
    t_redfac::Float64
    sigma1::Float64
    sigma2::Float64
    lmin::Float64
    lmax::Float64
    r_incfac::Float64
    epsnoprogf::Float64
    maxnoprogf::Int64
    # min scaling factors
    wmin::Float64
    scale::Bool
end

# convert a Vector{Int64} into a UnitRange if possible
function consec_range(v::Vector{Int64})
    if isempty(v)
        return v
    else
        sort!(v)
        if v[end] - v[1] + 1 == length(v)
            return v[1]:v[end]
        else
            return v
        end
    end
end
