This readme file was generated on 2025-02-25 by Yang Liu

GENERAL INFORMATION

Title of article:  Chip-scale reconfigurable carbon nanotube physical unclonable functions


Information about funding sources that supported the collection of the data: N/A


SHARING/ACCESS INFORMATION

Licenses/restrictions placed on the data: Attribution-NonCommercial-NoDerivs (CC BY-NC-ND)

Links to other publicly accessible locations of the data: N/A

Links/relationships to ancillary data sets: N/A

Was data derived from another source? No

Recommended citation for this dataset: N/A


DATA & FILE OVERVIEW

File List: Self-driving in Central Hong Kong

Relationship between files, if important: N/A

Additional related data collected that was not included in the current data package: N/A

Are there multiple versions of the dataset? No



METHODOLOGICAL INFORMATION

Description of methods used for collection/generation of data: We conduct the simulations using the network simulation systems OMNeT++ 6.0.2 and SUMO 1.19.0. OMNeT++ is responsible for detailed packet-level simulation of vehicle information interactions and data transfers. SUMO is used to create the traffic simulation, to generate the road network required for the simulation, and to represent the traffic demand. We choose a Hong Kong Central from OpenStreetMap as the simulation area, with the number of cars ranging from 10 to 100. Since SUMO requires the road network to be in its own format, the first step is to configure the desired road network on the OpenStreetMap web page and export it as an .osm file. The .osm file then needs to be converted to the .net.xml file format that SUMO accepts. Using the randomTrips.py utility provided by SUMO, the routing file .rou.xml can be generated and the simulation can then be configured using the .sumocfg file. Finally, the simulation is performed in OMNeT++ to simulate the exchange of information between vehicles based on physically unclonable functions.

For repuroducible run of XGBoost.py and GAN code.py, direct press "Reproducible Run" under the environment. The results show the output of those codes; For repuroducible run of self-driving car communications simulation,download the "Self-driving in Central Hong Kong.zip" file and unzip the file. Then, import it into OMNET++ 6.0.2 and rerun the code.

Methods for processing the data: See above.

Instrument- or software-specific information needed to interpret the data: OMNeT++ 6.0.2 and SUMO 1.19.0.

Standards and calibration information, if appropriate: N/A

Environmental/experimental conditions: Linux

Describe any quality-assurance procedures performed on the data: N/A

People involved with sample collection, processing, analysis and/or submission: Yang Liu


DATA-SPECIFIC INFORMATION FOR: 

Number of variables: car numbers

Number of cases/rows: 4

Variable List: N/A

Missing data codes: N/A

Specialized formats or other abbreviations used: N/A
