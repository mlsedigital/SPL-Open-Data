<p>
  <img src="./assets/mlse-banner.jpeg">
</p>

# SPL Open Data
The SPL Open Data repository acts as a collection of biomechanics datasets collected by Maple Leaf Sports & Entertainment's (MLSE) Sport Performance Lab (SPL) in Toronto, Ontario, Canada. Through the open-sourcing of this data, SPL's goal is to provide raw markerless motion capture data typically used by sports biomechanists to the general public in an effort to improve data equity and analytical biomechanical skills in the community.

## Research Realm: Fatigue and Kinetic Chain


#### TODO: SpellCheck



Kinetic Chains are a more mechanical interpretation of human movememnt. It considers how the movements of certain body parts are influenced and influence the movements of other body parts through the joints that connect them. Literature classifies movement into 2 types of kinetic chains: open and closed. 

Open chains describe movement that involves successively connected body parts (shoulder-elbow-wrist-fingers) where the final segment (in this example finger) can move freely. This form of chains primarily involve movement along one axis, and is defined 
when bodyparts do not move simultaneously. 

Closed chains describe movements defines the opposite, movement where the previous segments greatly limit the movement of the final segment (ankle-knee-hip) where the movement of different body segments happen concurrently rather sequentially eg. Squat where the torso and the femur move concurrently to the desired position. This requires joints and muscles to contract and move together to achieve the desired movement pattern. 

This study hopes to leverage the SPL Open Data (TODO: Add link or maybe not) to understand how repeated shots and fatigue impact these 2 qualities. We aim to study this through the following process (Subject to heavy changes):

### Segment the Shooting Process:
1. Break down each shot into non-overlapping phases:
  - Preparation/Set-Up
  - Downward Movement
  - Push-Off
  - Shooting/Release
  - Follow-Through

### Identify Kinematic Chain Status:

Classify body segments as either open or closed kinematic chains for each phase.
- Lower body: Closed during push-off (contact with the ground).
- Upper body: Open during release (free movement).

### Bayesian Modeling:

1. Priors: Model joint angles, angular velocities, and kinematic chain statuses for each phase.
2. Likelihood: Define how observed data (angles/velocities) arise from phase-specific priors, adjusting for shot number (fatigue).
3. Posterior Inference: Use MCMC (via PyMC3/Stan) to infer the posterior distribution of kinematic chain activations and joint
movements across phases.

### Variables
1. Joint Angles: Knee, hip, shoulder, elbow, wrist, ankle (both limbs).
2. Angular Velocity/Acceleration: Calculated from joint angles across phases.
3. Kinematic Chain Status: Open/closed status for upper and lower body during each phase.

### Methodology
1. Segmentation: Use key markers (e.g., knee, wrist) to identify the boundaries between shooting phases.
2. Open/Closed Chain Identification: Detect foot contact (closed) or free limb movement (open) based on marker data for each phase.
3. Bayesian Inference: 
    - Hierarchical Bayesian model to capture phase-specific distributions for joint kinematics. Account for shot number (fatigue proxy) as a covariate to model fatigue’s impact on joint angles, velocities, and kinematic chain activations.
4. Visualization
  - Phase-specific Joint Angle Distributions: Visualize how joint angles (e.g., elbow, wrist) shift across phases and fatigue levels.
  - Posterior Distribution Plots: Show Bayesian credible intervals for kinematic chain activations and velocities.
  - Interactive Dashboard (PyShiny/Dash): Explore phase-wise results, segment-wise kinematic changes, and fatigue progression.




## File Structure 

TODO: Update the file structure

Assuming that you have cloned the repository, the naming convention of the files is as follows. The data in this repository is structured in the following tree:

```
[sport]/
├─ [action_type]/
│  ├─ participant_information.json
│  ├─ [participant_id]/
│  │  ├─ trial_data/
|  |     ├─ [trial_id].json
```

where, trial data is unique to each individual participant and anonymized, demographic information relating to all participants is referenced in the `participant_information.json` file.

## Data summary

*Last updated: September 2024*

Each "Action" will typically feature a corresponding README file, with relevant documentation.

<center>

| Sport | Action | Participants | Trials (total) |
| :-: | :-: | :-: | :-: |
| Basketball | [Free throw](./basketball/freethrow/) | 1 | 125

</center>


# Data Collection & Participant Release
The datasets in this repository have been collected using SPL's internal biomechanics technology and the participants within them are non-professional athletes having given fully informed consent for the release of their movement capture data.

## License

License
CC BY-NC-SA 4.0

Attribution-NonCommercial-ShareAlike 4.0 International

https://creativecommons.org/licenses/by-nc-sa/4.0/

Detailed information can be found in LICENSE.