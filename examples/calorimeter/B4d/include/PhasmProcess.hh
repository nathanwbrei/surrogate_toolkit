
#pragma once
#include <G4ParticleChangeForLoss.hh>
#include <G4WrapperProcess.hh>
#include <surrogate_builder.h>

class PhasmProcess : public G4WrapperProcess {
public:
  enum class MethodToSurrogate {
    AlongStepDoIt,
    AtRestDoIt,
    PostStepDoIt,
    AlongStepGetPIL,
    AtRestGetPIL,
    PostStepGetPIL
  };

private:
  MethodToSurrogate m_method_to_surrogate = MethodToSurrogate::AlongStepDoIt;
  std::vector<G4ParticleDefinition *> m_particles_to_surrogate;

  static inline phasm::Surrogate AlongStepDoItSurrogate =
      phasm::SurrogateBuilder()
          .set_model("phasm-torch-plugin", "")
          .local<G4Track>("track")
          .accessor<double>(&G4Track::GetKineticEnergy,
                            &G4Track::SetKineticEnergy)
          .primitive("kineticEnergy", phasm::IN)
          .end()
          .end()
          .local<G4Step>("step")
          .accessor<double>(&G4Step::GetTotalEnergyDeposit,
                            &G4Step::SetTotalEnergyDeposit)
          .primitive("totalEnergyDeposit", phasm::IN)
          .end()
          .accessor<double>(&G4Step::GetNonIonizingEnergyDeposit,
                            &G4Step::SetNonIonizingEnergyDeposit)
          .primitive("nonIonizingEnergyDeposit", phasm::IN)
          .end()
          .end()
          .local<G4ParticleChangeForLoss>("particleChange")
          .accessor<double>(&G4ParticleChangeForLoss::GetCharge,
                            &G4ParticleChangeForLoss::ProposeCharge)
          .primitive("charge", phasm::OUT)
          .end()
          .accessor<CLHEP::Hep3Vector>(
              [](G4ParticleChangeForLoss *l) {
                return l->GetMomentumDirection();
              },
              [](G4ParticleChangeForLoss *l, CLHEP::Hep3Vector v) {
                l->ProposeMomentumDirection(v);
              })
          .accessor<double>(&CLHEP::Hep3Vector::x, &CLHEP::Hep3Vector::setX)
          .primitive("momentumDirectionX", phasm::OUT)
          .end()
          .accessor<double>(&G4ThreeVector::y, &G4ThreeVector::setY)
          .primitive("momentumDirectionY", phasm::OUT)
          .end()
          .accessor<double>(&G4ThreeVector::z, &G4ThreeVector::setZ)
          .primitive("momentumDirectionZ", phasm::OUT)
          .end()
          .end()
          .end()
          .finish();

public:
  PhasmProcess(G4VProcess *underlying, MethodToSurrogate method);

  virtual G4VParticleChange *PostStepDoIt(const G4Track &track,
                                          const G4Step &stepData);

  virtual G4VParticleChange *AlongStepDoIt(const G4Track &track,
                                           const G4Step &stepData);

  virtual G4VParticleChange *AtRestDoIt(const G4Track &track,
                                        const G4Step &stepData);

  virtual G4double AlongStepGetPhysicalInteractionLength(
      const G4Track &track, G4double previousStepSize,
      G4double currentMinimumStep, G4double &proposedSafety,
      G4GPILSelection *selection);

  virtual G4double
  AtRestGetPhysicalInteractionLength(const G4Track &track,
                                     G4ForceCondition *condition);

  virtual G4double
  PostStepGetPhysicalInteractionLength(const G4Track &track,
                                       G4double previousStepSize,
                                       G4ForceCondition *condition);
};
