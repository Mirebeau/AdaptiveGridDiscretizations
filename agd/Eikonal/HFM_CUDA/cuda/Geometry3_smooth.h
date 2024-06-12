#pragma once
// Copyright 2024 Jean-Marie Mirebeau, ENS Paris-Saclay, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/**
 This file implements a smooth variant of Selling's three-dimensional decomposition, still exploratory and unpublished. The two-dimensional counterpart is published in
 Bonnans, F., Bonnet, G. & Mirebeau, J.-M. Monotone discretization of anisotropic differential operators using Voronoi’s first reduction. Constructive Approximation 1–61 (2023).
 */


#include "Geometry3.h"
#include "GeometryT.h"
#include "Geometry6/Smooth3_data.h"
#include "Dense2.h"

/// Smooth variant of Selling's decomposition
const int smooth3_decomp_order = 6; // TUNING PARAMETER. Adjust smoothness
namespace smooth {
const int decompdim = 13; // Enough, we conjecture, to achieve a smooth decomposition.
const int rdim = decompdim-symdim;
// Number of superbases attaining the first two energy levels in Voronoi's reduction, worst case (Id)
const int sb_nmin1 = 16; const int sb_nmin2 = 16 + 36; // First and second level
const int BadIndex = int(-2e9);

/**Returns the primitive of (1-x^2)^order, vanishing at 0.*/
template<typename T> T _heaviside_helper(T x){
    const int order = smooth3_decomp_order;
    if(order==0) return x;
    const T x2 = x*x; const T x3 = x*x2;
    if(order<=2) return (x-x3*Scalar(1./3));
    const T x5 = x2*x3;
    if(order<=4) return (x-x3*Scalar(2./3)+x5*Scalar(1./5));
    const T x7 = x2*x5;
    if(order<=6) return (x-x3+x5*Scalar(3./5)-x7*Scalar(1./7));
}
const Scalar _heaviside_norm = 1./_heaviside_helper<Scalar>(1);

/**Regularized heaviside function, with a transition over [-1,1], with C^k regularity*/
template<typename T> T heaviside(T x){
    if(x <= -1) return 0;
    if(x >= 1)  return 1;
    return 0.5+0.5*_heaviside_helper(x) * _heaviside_norm;
}

Scalar cutoff(Scalar energy, Scalar emin, Scalar emax){
    const Scalar r = 0.5; // TUNING PARAMETER. Adjust cutoff. Needs 0 < r <= 1.
    const Scalar ediff = r*(emax-emin);
    const Scalar x = (emin+ediff-energy)/ediff;
    return heaviside(2*x-1);
}

template<typename T> void
_energy_ext_objective(const Scalar energy_val[sb_nmin2], const T emin, const T emax, T val[2]){
    const Scalar r = 0.5; // TUNING PARAMETER. Adjust cutoff. Needs 0 < r <= 0.5
    const T ediff = r*(emax-emin);
    val[0] = -0.5; val[1] = -16.5;
    for(int i=0; i<sb_nmin2; ++i){
        val[0] += heaviside((emin-energy_val[i])/ediff);
        val[1] += heaviside((emax-energy_val[i])/ediff);
    }
}
const int energy_ext_nitermax = 10;
/**Get approximately the 0th and 16th energy, in a smooth way**/
void energy_ext(const Scalar energy_val[sb_nmin2], Scalar & emin, Scalar & emax){
    typedef Dense1<Scalar,2> T;
    T emix[2], val[2]; // emin and emax, with AD, to be updated
    T::Identity(emix);
    emix[0].a = energy_val[0];
    emix[1].a = energy_val[sb_nmin1];
    Scalar direction[2];
    for(int i=0; i<energy_ext_nitermax; ++i){
        _energy_ext_objective(energy_val,emix[0],emix[1],val);
        T::solve(val, direction);
//        std::cout ExportVarArrow(emix[0].a) ExportVarArrow(emix[1].a)
//        ExportVarArrow(val[0].a) ExportVarArrow(val[1].a)
//        ExportArrayArrow(direction) << std::endl;
        for(int j=0; j<2; ++j){emix[j]+=direction[j];}
        // TODO : check that objective is decreased, if not reduce step
    }
    emin = emix[0].a;
    emax = emix[1].a;
}

template<int m,int n,typename Ta, typename Tx, typename Tout> void
tdot_av(const Ta a[m][n], const Tx x[m], Tout out[n]){
    for(int i=0; i<n; ++i){
        out[i] = 0;
        for(int j=0; j<m; ++j) out[i] += a[j][i]*x[j];}
}

/**Extend a family of rdim coefficients, so as to obtain a decomposition of the given matrix m. */
template<typename T> void
_decom_optim_complement(const Scalar m[symdim],
                        const Scalar iA[symdim][symdim], const Scalar rA[rdim][symdim],
                        T icoef[symdim], const T rcoef[rdim]){
    T err[symdim];
    tdot_av<rdim,symdim>(rA, rcoef, err); // Reconstruction using the last coefficients
    //    std::cout ExportArrayArrow(err)  << std::endl;
    //    for(int i=0; i<rdim; ++i) std::cout << rcoef[i] << ","; std::cout << std::endl;
    for(int i=0; i<symdim; ++i) err[i] = m[i] - err[i]; // Reconstruction error
    tdot_av<symdim,symdim>(iA, err, icoef); // Compensation using the first coefficients
    //    for(int i=0; i<symdim; ++i) std::cout << icoef[i] << ","; std::cout << std::endl;
}

/**The objective function defining the modified Selling decomposition, with weights mu.**/
template<typename T, typename Tout> Tout
_decomp_optim_objective(const T icoef[symdim], const T rcoef[rdim], const Scalar mu[decompdim]){
    Tout obj(0);
    const Scalar r = 0.1; // TUNING PARAMETER. Adjust objective vs penalization.
    for(int i=0;i<decompdim; ++i){
        const Tout lambda = Tout(i<symdim ? icoef[i] : rcoef[i-symdim]);
        if(mu[i]<=0) {obj += lambda*lambda; continue;} // These values remain at zero
        obj += (0.5*r/mu[i])*lambda*lambda - (r*mu[i]) * log(lambda) - lambda;
    }
    return obj;
};

const int _decomp_optim_nitermax = 20;
const int _decomp_optim_nsplit = 5;
/**Decompose D over the given offsets, with the given weights, see modified optimization problem*/
void _decomp_optim(const Scalar m[symdim],
                   const int offset_index[decompdim], const Scalar offset_weight[decompdim],
                   Scalar coef[__restrict__ decompdim]){
    // Normalize the matrix which is to be decomposed
    const Scalar sdet = pow(det_m(m),Scalar(1./3));
    Scalar m1[symdim]; // Normalized matrix, unit determinant
    GeometryT<symdim>::mul(Scalar(1)/sdet,m,m1);
    
    // Preprocess offsets
    Scalar _iA[symdim][symdim]; // Will be inverted. Inverse has integer entries, but still...
    Scalar rA[rdim][symdim];
    for(int i=0; i<symdim; ++i){self_outer_v(_smooth3_uoffset[offset_index[i]],_iA[i]);}
    for(int i=0; i<rdim;   ++i){self_outer_v(_smooth3_uoffset[offset_index[symdim+i]], rA[i]);}
    Scalar iA[symdim][symdim];
    GeometryT<symdim>::inv_a(_iA, iA);
    
    //    GeometryT<symdim>::show_a(iA);
    
    // Build an initial guess
    Scalar m0[symdim]; GeometryT<symdim>::zero(m0);
    for(int i=0; i<symdim; ++i) GeometryT<symdim>::madd(offset_weight[i],_iA[i], m0);
    for(int i=0; i<rdim;   ++i) GeometryT<symdim>::madd(offset_weight[symdim+i],rA[i], m0);
    Scalar im1[symdim]; GeometryT<ndim>::inv_m(m1,im1);
    const Scalar r = 1/scal_mm(m0, im1);
//    std::cout ExportVarArrow(r) ExportArrayArrow(m0) ExportArrayArrow(im1) ExportArrayArrow(m1) << std::endl;
    Scalar rcoef[rdim], icoef[symdim];
    for(int i=0; i<rdim; ++i) rcoef[i] = offset_weight[symdim+i]*r;
    for(int i=0; i<10; ++i){
        _decom_optim_complement(m1,iA,rA,icoef,rcoef);
        bool neg=false; for(int j=0; j<symdim; ++j) neg |= icoef[j]<0;
        if(neg) GeometryT<rdim>::mul(0.5,rcoef);
        else break;
    }
//    rcoef[0]=0.07723756348464; rcoef[1]=0.00771458922288; rcoef[2]=0.00197253024265; rcoef[3]=0.00011396434296;
//    std::cout ExportArrayArrow(rcoef) << std::endl;
    
    // Run the damped Newton solver to compute the optimal decomposition
    typedef Dense1<Scalar,rdim> T1;
    typedef Dense2<Scalar,rdim> T2;
    T1 icoef1[decompdim], rcoef1[rdim];
    for(int i=0; i<rdim; ++i) {rcoef1[i] = rcoef[i]; rcoef1[i].v[i]=1;}
    Scalar dir[rdim]; // Descent direction
    Scalar obj_prev = INFINITY;
    for(int i=0; i<_decomp_optim_nitermax; ++i){
        // Damped Newton method. TODO : introduce stopping criterion and early abort (?)
        _decom_optim_complement(m1,iA,rA,icoef1,rcoef1);
        T2 obj = _decomp_optim_objective<T1,T2>(icoef1,rcoef1,offset_weight);
//        std::cout ExportVarArrow(obj.a) << std::endl;
        // Adjust the gradient descent step, if objective was not decreased
        Scalar step=1;
        for(int j=0; !(obj<obj_prev) && j<_decomp_optim_nsplit; ++j){
            step /= 2;
            GeometryT<rdim>::madd(step,dir,rcoef1);
            _decom_optim_complement(m1,iA,rA,icoef1,rcoef1);
            obj = _decomp_optim_objective<T1,T2>(icoef1,rcoef1,offset_weight);
//            std::cout ExportVarArrow(step) ExportVarArrow(obj.a) << std::endl;
        }
        obj.solve_stationnary(dir);
        // Optimization opportunity : abort if the magnitude of dir approaches machine precision.
        // (Not difficult since weights and m1 are normalized, hence objective and solution are O(1))
        //obj.showself();
//        std::cout ExportArrayArrow(dir) << std::endl;
//        for(int j=0; j<rdim; ++j) std::cout << rcoef1[j].a << ","; std::cout << std::endl;
        GeometryT<rdim>::sub(dir,rcoef1);
    }
    for(int i=0; i<rdim; ++i) coef[symdim+i] = rcoef1[i].a;
    _decom_optim_complement(m1,iA,rA,coef,coef+symdim);
    GeometryT<decompdim>::mul(sdet,coef);
}

void decomp_m(const Scalar m[symdim],
              Scalar weights[__restrict__ decompdim],OffsetT offsets[__restrict__ decompdim][ndim]){
    OffsetT sb[ndim+1][ndim]; obtusesuperbase_m(m, sb);
//    OffsetT sb[ndim+1][ndim] = {{ 1, 1,-1},{ 0,-1, 0},{-1, 0, 0},{ 0, 0, 1}};
    Scalar m_ref[symdim]; // Unimodular transformation puts m into fundamental domain for GL3Z.
    tgram_am(sb, m, m_ref);
    
//    std::cout ExportArrayArrow(m_ref) << std::endl;
    // Find and sort the energy_n neighbor superbases with minimal energies
    Scalar energy_val[sb_nmin2];
    int energy_index[sb_nmin2];
    for(int i=0; i<sb_nmin2; ++i){energy_val[i]=INFINITY; energy_index[i]=BadIndex;}
    for(int i=0; i<_smooth3_nneigh; ++i){
        const Scalar e = GeometryT<symdim>::scal(m_ref,_smooth3_energy[i]); // Not scal_mm
//        if(e<7){std::cout ExportVarArrow(e) ExportVarArrow(i) << std::endl;}
        if(e>=energy_val[sb_nmin2-1]) continue; // Not a candidate for retained perfect forms
        for(int j=sb_nmin2-2; j>=-1; --j){
            if(j==-1 || e>=energy_val[j]){energy_val[j+1]=e; energy_index[j+1]=i; break;}
            else {energy_val[j+1]=energy_val[j]; energy_index[j+1]=energy_index[j];}
        } // for j
    } // for i
            
    // Compute the weights associated to the superbases
    Scalar emin,emax;
    energy_ext(energy_val,emin,emax);
    Scalar rho[sb_nmin1], rho_sum=0;
    for(int i=0; i<sb_nmin1; ++i) {rho[i] = cutoff(energy_val[i],emin,emax); rho_sum += rho[i];}
    const Scalar rho_isum = 1./rho_sum; Int rho_n = 0;
    for(int i=0; i<sb_nmin1; ++i) {rho[i] *= rho_isum; rho_n += rho[i]>0; }
//    std::cout ExportVarArrow(emin) ExportVarArrow(emax) << std::endl;
    
/*    std::cout
    ExportArrayArrow(rho)
//    ExportArrayArrow(energy_index)
    //ExportVarArrow(rho_sum)
    //ExportArrayArrow(energy_val)
    << std::endl;*/
    
//    GeometryT<decompdim>::show_v(energy_val); std::cout << "Energies\n";
//    GeometryT<decompdim>::show_v(energy_index); std::cout << "Indices\n";
    
    
    
    // Obtain the offsets associated to the superbases, and the corresponding weights
    const int offset_free = _smooth3_nuoffset;
    int offset_index[decompdim];
    Scalar offset_weight[decompdim];
    for(int i=0; i<decompdim; ++i) {offset_index[i] = offset_free; offset_weight[i] = 0;}
    for(int i=0; i<sb_nmin1; ++i){
        if(i>=rho_n) break; // Discard superbases with null weights
        for(int j=0; j<6; ++j){ // Consider all offsets of this superbase
            const int ioffset = _smooth3_ioffset[energy_index[i]][j];
//            if(i==0){std::cout ExportVarArrow(ioffset);}
            for(int k=0; k<decompdim; ++k){ // See if this offset was already registered
                // Note that if there are more than decompdim offsets, then the others will be
                // ignored. In that case, we expect (hope) their weights to be negligible.
                if(offset_index[k]==ioffset){offset_weight[k]+=rho[i]; break;}
                if(offset_index[k]==offset_free){offset_weight[k]=rho[i]; offset_index[k]=ioffset; break;}
            }
        }
    }
    
//    std::cout ExportArrayArrow(offset_weight)
//    ExportArrayArrow(offset_index) << std::endl;
    
//    GeometryT<ndim>::show_a<decompdim>(offsets);
    
    // Compute the coefficients, map the offsets using inverse superbase
    _decomp_optim(m_ref, offset_index, offset_weight, weights);
    OffsetT isb[ndim][ndim]; // Comatrix (+- transposed inverse) of the superbase transformation
    for(int i=0; i<ndim; ++i)
        for(int j=0; j<ndim; ++j)
            isb[j][i]=
            sb[(i+1)%3][(j+1)%3]*sb[(i+2)%3][(j+2)%3]-
            sb[(i+1)%3][(j+2)%3]*sb[(i+2)%3][(j+1)%3];
    for(int i=0; i<decompdim; ++i)
        dot_av(isb,_smooth3_uoffset[offset_index[i]],offsets[i]);
} // decomp_m

} // smooth
