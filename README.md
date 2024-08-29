<p>
  <img src="./assets/mlse-banner.jpeg">
</p>

# SPL Open Data
The SPL Open Data repository acts as a collection of biomechanics datasets collected by Maple Leaf Sports & Entertainment's (MLSE) Sport Performance Lab (SPL) in Toronto, Ontario, Canada. Through the open-sourcing of this data, SPL's goal is to provide raw markerless motion capture data typically used by sports biomechanists to the general public in an effort to improve data equity and analytical biomechanical skills in the community.

For more information on SPL, please visit the [lab's homepage](https://www.mlsedigital.com/innovation-initiatives/sport-performance-lab).

# Quick Start

Git clone this package using:

```
git clone https://github.com/mlsedigital/SPL-Open-Data.git
```

# File Structure

The data in this repository is structured in the following tree:

```
sport/
├─ action_type/
│  ├─ participant_information.json
│  ├─ participant_data/
│  │  ├─ trial_data
```

where, trial data is unique to each individual participant and anonymized, demographic information relating to all participants is referenced in the participant_information.json file.

# License

License
CC BY-NC-SA 4.0

Attribution-NonCommercial-ShareAlike 4.0 International

https://creativecommons.org/licenses/by-nc-sa/4.0/

Detailed information can be found in LICENSE.

# Data Collection & Participant Release
The datasets in this repository have been collected using SPL's internal biomechanics technology and the participants within them are non-professional athletes having given fully informed consent for the release of their movement capture data.

# Inspirations
This package is inspired by Driveline Baseball's [OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics), one of the largest collections of open-source baseball and high performance biomechanics datasets, and with these datasets, we hope to also inspire the future release of biomechanical data from other organizations.
