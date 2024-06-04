#pragma once
// Copyright 2024 Jean-Marie Mirebeau, ENS Paris-Saclay, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/**
 This file implements a smooth variant of Selling's three-dimensional decomposition, still exploratory and unpublished. The two-dimensional counterpart is published in
 Bonnans, F., Bonnet, G. & Mirebeau, J.-M. Monotone discretization of anisotropic differential operators using Voronoi’s first reduction. Constructive Approximation 1–61 (2023).
 */

#include "Geometry3.h"
#include "GeometryT.h"

namespace smooth2 { // alt because this is a second attempt...
const int decompdim = 13; // // 37 is a guaranteed upper bound, but we conjecture that 13 is sufficient

// The following candidate superbases *may* satisfy E^3 <= Emin^3 + det(D)
const int ntot_sb = 127;
const int8_t tot_energies[ntot_sb][symdim] =
{{1,1,1,1,1,1},{1,1,1,1,1,2},{1,1,1,1,2,1},{1,1,1,1,2,3},{1,1,1,1,3,2},{1,1,1,1,3,3},{1,1,1,2,1,1},{1,1,1,2,1,3},{1,1,1,2,3,1},{1,1,1,2,3,3},{1,1,1,3,1,2},{1,1,1,3,1,3},{1,1,1,3,2,1},{1,1,1,3,2,3},{1,1,1,3,3,1},{1,1,1,3,3,2},{1,1,2,1,1,1},{1,1,2,1,3,1},{1,1,2,1,3,3},{1,1,2,1,5,3},{1,1,2,3,1,1},{1,1,2,3,1,3},{1,1,2,5,1,3},{1,1,3,1,2,1},{1,1,3,1,3,1},{1,1,3,1,3,2},{1,1,3,1,5,2},{1,1,3,1,5,3},{1,1,3,1,6,3},{1,1,3,2,1,1},{1,1,3,3,1,1},{1,1,3,3,1,2},{1,1,3,5,1,2},{1,1,3,5,1,3},{1,1,3,6,1,3},{1,2,1,1,1,1},{1,2,1,1,1,3},{1,2,1,1,3,1},{1,2,1,3,1,3},{1,2,1,3,1,5},{1,2,1,3,3,1},{1,2,1,3,5,1},{1,2,3,1,1,1},{1,2,3,1,3,1},{1,2,3,3,1,1},{1,2,5,3,1,1},{1,3,1,1,1,2},{1,3,1,1,1,3},{1,3,1,1,2,1},{1,3,1,1,3,1},{1,3,1,2,1,3},{1,3,1,2,1,5},{1,3,1,2,3,1},{1,3,1,2,5,1},{1,3,1,3,1,5},{1,3,1,3,1,6},{1,3,1,3,5,1},{1,3,1,3,6,1},{1,3,2,1,1,1},{1,3,2,1,1,3},{1,3,2,1,3,1},{1,3,3,1,1,1},{1,3,3,1,1,2},{1,3,3,1,2,1},{1,3,3,2,1,1},{1,3,5,2,1,1},{1,3,5,3,1,1},{1,3,6,3,1,1},{1,5,2,1,1,3},{1,5,3,1,1,2},{1,5,3,1,1,3},{1,6,3,1,1,3},{2,1,1,1,1,1},{2,1,1,1,1,3},{2,1,1,1,3,3},{2,1,1,1,3,5},{2,1,1,3,1,1},{2,1,1,3,3,1},{2,1,1,5,3,1},{2,1,3,1,1,1},{2,1,3,1,3,1},{2,1,3,3,1,1},{2,1,5,1,3,1},{2,3,1,1,1,1},{2,3,1,1,1,3},{2,3,1,1,3,1},{2,5,1,1,3,1},{3,1,1,1,1,2},{3,1,1,1,1,3},{3,1,1,1,2,3},{3,1,1,1,2,5},{3,1,1,1,3,5},{3,1,1,1,3,6},{3,1,1,2,1,1},{3,1,1,3,1,1},{3,1,1,3,2,1},{3,1,1,5,2,1},{3,1,1,5,3,1},{3,1,1,6,3,1},{3,1,2,1,1,1},{3,1,2,1,1,3},{3,1,2,3,1,1},{3,1,3,1,1,1},{3,1,3,1,1,2},{3,1,3,1,2,1},{3,1,3,2,1,1},{3,1,5,1,2,1},{3,1,5,1,3,1},{3,1,6,1,3,1},{3,2,1,1,1,1},{3,2,1,1,1,3},{3,2,1,3,1,1},{3,3,1,1,1,1},{3,3,1,1,1,2},{3,3,1,1,2,1},{3,3,1,2,1,1},{3,5,1,1,2,1},{3,5,1,1,3,1},{3,6,1,1,3,1},{5,1,2,1,1,3},{5,1,3,1,1,2},{5,1,3,1,1,3},{5,2,1,3,1,1},{5,3,1,2,1,1},{5,3,1,3,1,1},{6,1,3,1,1,3},{6,3,1,3,1,1}};

const int ntot_offset=37;
const int8_t tot_offsets[ntot_offset][ndim] = {{1,0,0},{1,0,-1},{1,-1,0},{0,1,0},{0,1,-1},{0,0,1},{1,1,-1},{1,-1,-1},{2,0,-1},{2,-1,-1},{1,-1,1},{3,-1,-1},{2,-1,0},{0,1,1},{2,1,-1},{0,1,-2},{2,1,-2},{1,1,0},{1,1,-2},{0,2,-1},{2,-2,1},{2,-1,1},{1,1,1},{1,-1,2},{1,-2,1},{1,0,1},{1,2,-1},{2,-2,-1},{2,-1,-2},{1,-1,-2},{1,1,-3},{1,-3,1},{1,-2,-1},{1,-2,0},{1,0,-2},{1,2,-2},{1,-2,2}};

const uint8_t itot_offsets[ntot_sb][symdim] = {{0,1,2,3,4,5},{6,0,1,2,3,4},{0,1,2,7,3,5},{8,6,0,1,2,3},{9,0,1,2,7,3},{8,9,0,1,2,3},{0,1,10,2,4,5},{8,6,0,1,2,4},{9,0,1,2,7,5},{11,8,9,0,1,2},{12,0,1,10,2,4},{8,12,0,1,2,4},{12,0,1,10,2,5},{11,8,12,0,1,2},{12,9,0,1,2,5},{11,12,9,0,1,2},{6,0,1,3,4,5},{0,1,7,13,3,5},{14,8,6,0,1,3},{8,9,0,1,7,3},{0,1,10,4,15,5},{16,8,6,0,1,4},{8,12,0,1,10,4},{17,6,0,1,3,5},{17,0,1,13,3,5},{14,17,6,0,1,3},{17,0,1,7,13,3},{14,8,17,0,1,3},{8,17,0,1,7,3},{6,18,0,1,4,5},{18,0,1,4,15,5},{16,6,18,0,1,4},{18,0,1,10,4,15},{16,8,18,0,1,4},{8,18,0,1,10,4},{0,10,2,3,4,5},{6,0,2,19,3,4},{0,2,7,13,3,5},{12,20,0,10,2,4},{8,12,6,0,2,4},{21,12,0,10,2,5},{12,9,0,2,7,5},{17,6,0,3,4,5},{22,17,0,13,3,5},{0,23,10,4,15,5},{6,18,0,4,15,5},{0,10,2,24,3,4},{0,2,24,19,3,4},{25,0,10,2,3,5},{25,0,2,13,3,5},{20,0,10,2,24,4},{6,0,2,24,19,4},{21,25,0,10,2,5},{25,0,2,7,13,5},{12,20,0,2,24,4},{12,6,0,2,24,4},{21,12,25,0,2,5},{12,25,0,2,7,5},{25,0,10,3,4,5},{26,6,0,19,3,4},{22,25,0,13,3,5},{17,25,0,3,4,5},{26,17,6,0,3,4},{22,17,25,0,3,5},{25,0,23,10,4,5},{17,6,25,0,4,5},{25,0,23,4,15,5},{6,25,0,4,15,5},{0,10,24,19,3,4},{17,25,0,10,3,4},{26,17,0,19,3,4},{17,0,10,19,3,4},{1,2,7,3,4,5},{6,1,2,19,3,4},{9,27,1,2,7,3},{8,9,6,1,2,3},{1,10,2,4,15,5},{9,28,1,2,7,5},{12,9,1,10,2,5},{6,18,1,3,4,5},{1,7,29,13,3,5},{18,30,1,4,15,5},{17,6,1,13,3,5},{10,2,24,3,4,5},{2,24,31,19,3,4},{2,7,32,13,3,5},{25,10,2,13,3,5},{1,2,7,33,3,4},{1,2,33,19,3,4},{27,1,2,7,33,3},{6,1,2,33,19,3},{9,27,1,2,33,3},{9,6,1,2,33,3},{1,34,2,7,4,5},{1,34,2,4,15,5},{28,1,34,2,7,5},{1,34,10,2,15,5},{9,28,1,34,2,5},{9,1,34,10,2,5},{1,34,7,3,4,5},{35,6,1,19,3,4},{30,1,34,4,15,5},{18,1,34,3,4,5},{35,6,18,1,3,4},{1,34,7,29,3,5},{18,30,1,34,4,5},{6,18,1,34,3,5},{1,34,29,13,3,5},{6,1,34,13,3,5},{2,7,33,3,4,5},{2,33,31,19,3,4},{10,2,36,4,15,5},{2,24,33,3,4,5},{2,24,33,31,3,4},{2,7,33,32,3,5},{10,2,36,24,4,5},{10,2,24,33,3,5},{2,33,32,13,3,5},{10,2,33,13,3,5},{1,7,33,19,3,4},{18,1,34,7,3,4},{35,18,1,19,3,4},{34,2,7,4,15,5},{2,7,24,33,4,5},{2,36,24,4,15,5},{18,1,7,19,3,4},{2,7,24,4,15,5}};

/*
/// Satisfies f(0)=1, f(1)=0, smooth non-negative decreasing
Scalar cutoff(Scalar t) {return t>=Scalar(1) ? Scalar(0) : exp(Scalar(2)-Scalar(2)/(Scalar(1)-t));}
// Derivative of logarithm of cutoff
Scalar cutoff_dlog(Scalar t) {const Scalar s=1-t; return t>=Scalar(1) ? Scalar(0) : -Scalar(2)/(s*s)}
*/

void decomp_m(const Scalar m[symdim],
              Scalar weights[__restrict__ decompdim],
              OffsetT offsets[__restrict__ decompdim][ndim],
              Scalar relax=0.004, // Relaxation parameter for the modified Selling decomposition
              bool sb0 = false
){
    typedef GeometryT<symdim> GeoSym;
    OffsetT sb[ndim+1][ndim]; 
    obtusesuperbase_m(m, sb);
    if(sb0) canonicalsuperbase(sb); // Debug only
    Scalar m_ref[symdim]; // Unimodular transformation puts m into fundamental domain for GL3Z.
    tgram_am(sb, m, m_ref);
    const Scalar lambda[symdim] = {coef_m(m_ref,0,0)+coef_m(m_ref,0,1)+coef_m(m_ref,0,2),
        -coef_m(m_ref,0,2),-coef_m(m_ref,0,1),coef_m(m_ref,1,0)+coef_m(m_ref,1,1)+coef_m(m_ref,1,2),
        -coef_m(m_ref,1,2),coef_m(m_ref,2,0)+coef_m(m_ref,2,1)+coef_m(m_ref,2,2)};
    for(int i=0; i<symdim; ++i) assert(lambda[i]>=0);
    
    /* Get the restricted superbase candidates. We conjecture that there are 16 at most, which is
    attained in the case of the identity matrix. (ntot_sb = 127 is an upper bound) */
    const int nmax_sb = 16;
    int n_sb = 0;
    Scalar scores[nmax_sb];
    uint8_t i_sbs[nmax_sb];
    const Scalar energy0 = GeoSym::scal(lambda,tot_energies[0]); // Minimal energy
    const Scalar det = det_m(m_ref), energy0_3 = energy0*energy0*energy0;
    relax *= pow(det, Scalar(1./3));
    for(int i=0; i<ntot_sb; ++i){
        const Scalar energy = GeoSym::scal(lambda,tot_energies[i]);
        const Scalar energy_3 = energy*energy*energy, score = (energy_3-energy0_3)/det;
        assert(score>=0);
        if(score>=1) continue;
        assert(n_sb<nmax_sb);
        i_sbs[n_sb] = i;
        scores[n_sb] = score;
        ++n_sb;
    }
    
    // Compute a softmin for the superbases energies, using a Newton method
    const int nitermax_softmin = 10;
    Scalar softmin=0;
    for(int niter=0; niter<nitermax_softmin; ++niter){
        Scalar val = 0, dval = 0;
        for(int n=0; n<n_sb; ++n){
            const Scalar t = scores[n]-softmin;
            if(t>=1) continue;
            const Scalar s = 1/(1-t); // The cutoff function is exp(2-2/(1-t)) if t<1, else 0
            const Scalar cutoff = exp(2-2*s);
            const Scalar dcutoff = cutoff * 2*s*s; // (negative) derivative of cutoff
            val+=cutoff;
            dval+=dcutoff;
        }
        softmin -= (val-1)/dval;
    }
    
    // Compute the weights associated to the offsets
    int8_t i_offsets[decompdim];
    Scalar w_offsets[decompdim];
    {// The first 6 offset are associated to the first superbase (the Selling obtuse one)
        assert(abs(scores[0])<1e-5); // Should be zero.
        const Scalar t = scores[0]-softmin, s = 1/(1-t), cutoff = exp(2-2*s);
        for(int i=0; i<symdim; ++i){
            i_offsets[i]=i;
            w_offsets[i]=cutoff;
        }
    }
    int n_offsets = 6;
    for(int n=1; n<n_sb; ++n){
        const Scalar t = scores[n]-softmin;
        if(t>=1) continue;
        const Scalar s = 1/(1-t), cutoff = exp(2-2*s);
        const uint8_t i_sb = i_sbs[n];
        for(int i=0; i<symdim; ++i){
            const int8_t i_offset = itot_offsets[i_sb][i];
            // Check wether this offset was already registered
            int k=0;
            for(; k<n_offsets; ++k){
                if(i_offsets[k]==i_offset) {w_offsets[k]+=cutoff; break;}
            }
            if(k==n_offsets){
                assert(n_offsets<decompdim);
                i_offsets[n_offsets] = i_offset;
                w_offsets[n_offsets] = cutoff;
                n_offsets+=1;
            }
        }
    }
#ifndef NDEBUG
    for(int n=n_offsets; n<decompdim; ++n) {w_offsets[n]=0; i_offsets[n]=-128;}
    Scalar w_offsets_sum=0;
    for(int n=0; n<n_offsets; ++n) w_offsets_sum += w_offsets[n];
    assert(abs(w_offsets_sum-symdim)<1e-5);
#endif
    
    // Prepare for the Newton method
    const int tensordim = (symdim*(symdim+1))/2;
//    Scalar offsets[decompdim][ndim];
    int8_t offsets_[decompdim][ndim];
    int16_t offsets_m[decompdim][symdim];
    int16_t offsets_mm[decompdim][tensordim];
    for(int n=0; n<n_offsets; ++n){
        const int8_t * o = tot_offsets[i_offsets[n]];
        copy_vV(o,offsets_[n]);
        self_outer_v(o,offsets_m[n]);
        offsets_m[n][1]*=int8_t(2); offsets_m[n][3]*=int8_t(2); offsets_m[n][4]*=2;//Frobenius dual
        GeoSym::self_outer(offsets_m[n],offsets_mm[n]);
    }
    
    // Now, run a Newton method in dual space
    const int nitermax_dual = 12;
    Scalar m_opt[symdim] = {1.,1./2,1.,1./2,1./2,1.};
    for(int niter=0; niter<nitermax_dual; ++niter){
        Scalar obj = scal_mm(m_ref, m_opt);
        Scalar dobj[symdim]; copy_mM(m_ref,dobj);
        dobj[1] *= 2; dobj[3]*=2; dobj[4]*=2;
        Scalar ddobj[tensordim]; GeometryT<tensordim>::zero(ddobj);
        for(int n=0; n<n_offsets; ++n){
            const Scalar t = (1 - GeoSym::scal(m_opt,offsets_m[n])) / relax;
            // Compute the barrier function, and its first and second order derivatives
            const Scalar t2 = t/2, sqt2 = sqrt(1+t2*t2);
            const Scalar ddB = 0.5 + 0.5*t2/sqt2;
            const Scalar dB = t2 + sqt2;
            const Scalar B = t*dB - (dB*dB/2 - log(dB));
            // Add to the objective and derivatives
            obj += relax*w_offsets[n]*B;
            GeoSym::madd(-w_offsets[n]*dB,offsets_m[n],dobj);
            GeometryT<tensordim>::madd(w_offsets[n]*ddB/relax,offsets_mm[n],ddobj);
        } // for n_offsets
        // Compute the descent direction
        Scalar descent[symdim];
        GeoSym::solve_mv(ddobj,dobj,descent);
        GeoSym::sub(descent,m_opt); // Using a time step of 1. Adaptive timesteps may be needed.
    } // for nitermax_dual
    
    // Compute the decomposition weights using the optimality conditions
    for(int n=0; n<n_offsets; ++n){
        const Scalar t = (1 - GeoSym::scal(m_opt,offsets_m[n])) / relax;
        const Scalar t2 = t/2, sqt2 = sqrt(1+t2*t2), dB = t2 + sqt2;
        weights[n] = w_offsets[n] * dB;
    } // for n_offsets
    
    // Compute the offsets using a change of coordinates
    OffsetT isb[ndim][ndim]; // Comatrix (+- transposed inverse) of the superbase transformation
    for(int i=0; i<ndim; ++i)
        for(int j=0; j<ndim; ++j)
            isb[j][i]=
            sb[(i+1)%3][(j+1)%3]*sb[(i+2)%3][(j+2)%3]-
            sb[(i+1)%3][(j+2)%3]*sb[(i+2)%3][(j+1)%3];
    for(int n=0; n<n_offsets; ++n)
        dot_av(isb,offsets_[n],offsets[n]);
    
    for(int n=n_offsets; n<decompdim; ++n){weights[n]=0; zero_V(offsets[n]);}
}


} // smooth2
