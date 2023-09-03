
#pragma once
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
          .set_callmode(phasm::CallMode::DumpTrainingData)
          .local_primitive<double>("x", phasm::IN)
          .local_primitive<double>("y", phasm::OUT)
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
