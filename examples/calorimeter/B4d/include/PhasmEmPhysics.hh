
#pragma once

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

class PhasmEmPhysics : public G4VPhysicsConstructor {
public:
  explicit PhasmEmPhysics(G4int ver = 1, const G4String &name = "");
  ~PhasmEmPhysics() override;

  void ConstructParticle() override;
  void ConstructProcess() override;
};
