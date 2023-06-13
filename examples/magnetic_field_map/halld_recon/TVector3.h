
// @(#)root/physics:$Id$
// Author: Pasha Murat, Peter Malzacher   12/02/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TVector3
#define ROOT_TVector3

using Double_t = double;
#include <cmath>

class TVector3 {
    double fX, fY, fZ;

public:
    inline Double_t x()  const;
    inline Double_t y()  const;
    inline Double_t z()  const;

    inline void SetXYZ(Double_t x, Double_t y, Double_t z);

    inline Double_t Perp() const;
    // The transverse component (R in cylindrical coordinate system).

    inline Double_t Perp2() const;
    // The transverse component squared (R^2 in cylindrical coordinate system).

};

using DVector3 = TVector3;


inline void TVector3::SetXYZ(Double_t xx, Double_t yy, Double_t zz) {
    fX = xx;
    fY = yy;
    fZ = zz;
}
inline Double_t TVector3::Perp2() const { return fX*fX + fY*fY; }

inline Double_t TVector3::Perp() const { return sqrt(Perp2()); }

inline Double_t TVector3::x()  const { return fX; }
inline Double_t TVector3::y()  const { return fY; }
inline Double_t TVector3::z()  const { return fZ; }

#endif