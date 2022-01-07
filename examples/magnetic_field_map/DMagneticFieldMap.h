
// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#ifndef SURROGATE_TOOLKIT_DMAGNETICFIELDMAP_H
#define SURROGATE_TOOLKIT_DMAGNETICFIELDMAP_H

#include "DVector3.h"

class DMagneticFieldMap{
public:

    DMagneticFieldMap(){}
    virtual ~DMagneticFieldMap(){}

    virtual void GetField(const DVector3 &pos,DVector3 &Bout) const=0;
    virtual void GetField(double x, double y, double z, double &Bx, double &By, double &Bz, int method=0) const = 0;
    virtual double GetBz(double x, double y, double z) const=0;

    virtual void GetFieldGradient(double x, double y, double z,
                                  double &dBxdx, double &dBxdy,
                                  double &dBxdz,
                                  double &dBydx, double &dBydy,
                                  double &dBydz,
                                  double &dBzdx, double &dBzdy,
                                  double &dBzdz) const = 0;
    virtual	void GetFieldBicubic(double x,double y,double z,
                                        double &Bx,double &By,double &Bz) const=0;

    virtual void GetFieldAndGradient(double x,double y,double z,
                                     double &Bx,double &By,
                                     double &Bz,
                                     double &dBxdx,
                                     double &dBxdy,
                                     double &dBxdz,
                                     double &dBydx,
                                     double &dBydy,
                                     double &dBydz,
                                     double &dBzdx,
                                     double &dBzdy,
                                     double &dBzdz) const = 0;


};

#endif //SURROGATE_TOOLKIT_DMAGNETICFIELDMAP_H
