#include "PhasmProcess.hh"

PhasmProcess::PhasmProcess(G4VProcess *underlying, MethodToSurrogate method)
    : m_method_to_surrogate(method) {
  pRegProcess = underlying; // Does NOT own underlying
  theProcessName = "Phasm" + underlying->GetProcessName();
  theProcessType = underlying->GetProcessType();
  theProcessSubType = underlying->GetProcessSubType();
}

G4double PhasmProcess::AlongStepGetPhysicalInteractionLength(
    const G4Track &track, G4double previousStepSize,
    G4double currentMinimumStep, G4double &proposedSafety,
    G4GPILSelection *selection) {

  if (m_method_to_surrogate == MethodToSurrogate::AlongStepGetPIL) {
    std::cout << "PHASM: Inside AlongStepGetPIL" << std::endl;
    return pRegProcess->AlongStepGetPhysicalInteractionLength(
        track, previousStepSize, currentMinimumStep, proposedSafety, selection);
  } else {
    return pRegProcess->AlongStepGetPhysicalInteractionLength(
        track, previousStepSize, currentMinimumStep, proposedSafety, selection);
  }
}

G4double
PhasmProcess::AtRestGetPhysicalInteractionLength(const G4Track &track,
                                                 G4ForceCondition *condition) {
  if (m_method_to_surrogate == MethodToSurrogate::AtRestGetPIL) {
    std::cout << "PHASM: Inside AtRestGetPIL" << std::endl;
    return pRegProcess->AtRestGetPhysicalInteractionLength(track, condition);
  } else {
  }
  return pRegProcess->AtRestGetPhysicalInteractionLength(track, condition);
}

G4double PhasmProcess::PostStepGetPhysicalInteractionLength(
    const G4Track &track, G4double previousStepSize,
    G4ForceCondition *condition) {
  if (m_method_to_surrogate == MethodToSurrogate::PostStepGetPIL) {
    std::cout << "PHASM: Inside PostStepGetPIL" << std::endl;
    return pRegProcess->PostStepGetPhysicalInteractionLength(
        track, previousStepSize, condition);
  } else {
    return pRegProcess->PostStepGetPhysicalInteractionLength(
        track, previousStepSize, condition);
  }
}

G4VParticleChange *PhasmProcess::AlongStepDoIt(const G4Track &track,
                                               const G4Step &stepData) {
  if (m_method_to_surrogate == MethodToSurrogate::AlongStepDoIt) {

    std::cout << "PHASM: Inside surrogate AlongStepDoIt" << std::endl;

    G4VParticleChange *result;

    AlongStepDoItSurrogate.bind_original_function(
        [&]() { result = pRegProcess->AlongStepDoIt(track, stepData); });
    AlongStepDoItSurrogate.bind_all_callsite_vars(
        const_cast<G4Track *>(&track), const_cast<G4Step *>(&stepData), result);
    AlongStepDoItSurrogate.call();
    return result;

  } else {
    return pRegProcess->AlongStepDoIt(track, stepData);
  }
}

G4VParticleChange *PhasmProcess::AtRestDoIt(const G4Track &track,
                                            const G4Step &stepData) {

  if (m_method_to_surrogate == MethodToSurrogate::AtRestDoIt) {
    std::cout << "PHASM: Inside AtRestDoIt" << std::endl;
    return pRegProcess->AtRestDoIt(track, stepData);
  } else {
    return pRegProcess->AtRestDoIt(track, stepData);
  }
}

G4VParticleChange *PhasmProcess::PostStepDoIt(const G4Track &track,
                                              const G4Step &stepData) {

  if (m_method_to_surrogate == MethodToSurrogate::PostStepDoIt) {
    std::cout << "PHASM: Inside PostStepDoIt" << std::endl;
    return pRegProcess->PostStepDoIt(track, stepData);
  } else {
    return pRegProcess->PostStepDoIt(track, stepData);
  }
}
