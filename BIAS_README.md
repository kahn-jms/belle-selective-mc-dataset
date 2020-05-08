# Bias quantification

Included in the dataset are event-level kinematic variables which describe the whole graph. These variables are derived from  reconstructed particles and can't be used as network input. However they can (should) be used during training as bias observables.

To quantify biases we are using the KL divergence between all true and true positive events. True events are labeled in the dataset. True positive events are true events which have also been classified as true events by the network.

To provide accurate data for specialized WGs to use in physics analysis, we want to keep the KL divergence below a certain value across all bias observables. This restriction combined with the highest speedup describes the performance of a certain model.

## KL divergence

To evaluate the bias overall, we use the Kullback-Leibler divergence between the expected (true events) and resulting distributions (positive) for decorrelation variables. Since there is still a selection after reconstruction, we can only use true positive events.

Kullback-Leibler divergence of Q from P: $D_{KL}(P\|Q)=-\sum_{x\in\chi}{P(x)log\left(\frac{Q(x)}{P(x)}\right)}$

The KL divergence should be below $KL_{div}<0.0015$. This value may change for different WGs depending on the results.

## Bias observables

| Bias observable | Description
|-|-
|nTracks|  number of tracks in the event
|abs(daughterDiffOf(0,1,mcDecayTime))|   Proper decay time difference Δt between daughter particles in ps.
|cosTheta|  momentum cosine of polar angle
cosToThrustOfEvent| cosine of the angle between the particle and the thrust axis of the even
|daughter(0,cosTheta)|   returns the $cos(\Theta)$ of the first daughter
|Q|  ???
|aplanarity| Event aplanarity, defined as the 3/2 of the third sphericity eigenvalue
|backwardHemisphereEnergy| Total energy the particles flying in the direction opposite to the thrust axis
|backwardHemisphereMass|   Invariant mass of the particles flying in the direction opposite to the thrust axis
|backwardHemisphere Momentum| Total momentum the particles flying in the direction opposite to the thrust axis
|backwardHemisphereX|  X component of the total momentum of the particles flying in the direciton opposite to the thrust axis
|backwardHemisphereY|  Y component of the total momentum of the particles flying in the direction opposite to the thrust axis
|backwardHemisphereZ|  Z component of the total momentum of the particles flying in the direction opposite to the thrust axis
|sphericity| Event sphericity, defined as the linear combination of the sphericity eigenvlaues S = (3/2)(lambda2+lambda3)
|thrust|    Event thrust
|cleoConeThrust(1-8)| Event shape variables
|foxWolframR(1-4)| Event shape variables
|missingEnergyofEventCMS|    The missing energy in CMS
|missingMomentum of EventCMS|  The magnitude of the missing momentum in CMS
|missingMomentumofEventCMS_Px|   The x component of the missing momentum in CMS
|missingMomentumofEventCMS_Py|   The y component of the missing momentum in CMS
|missingMomentumofEventCMS_Pz|   The z component of the missing momentum in CMS
|missingMomentumofEventCMS_theta|     The theta angle of the missing momentum in CMS
|visibleEnergyofEventCMS|    The visible energy in CMS
|missingMass2ofEvent| The missing mass squared
|missingMomentumofEvent| The magnitude of the missing momentum in lab
|missingMomentumofEvent_Px|  The x component of the missing momentum in lab
|missingMomentumofEvent_Py|  The y component of the missing momentum in lab
|missingMomentumofEvent_Pz|  The z component of the missing momentum in lab
|missingMomentumofEvent_theta|   The theta angle of the missing momentum of the event in lab
|totalPhotonEnergyofEvent|  The energy in lab of all the photons
