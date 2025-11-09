# BDSN-FDC
Repository for the Bayesian Decision Support Network for Fault Diagnosis and Correction (BDSN-FDC). The BDSN-FDC includes an ontology-driven Bayesian network approach to fault diagnosis and correction in manufacturing

## Repository Overview

This repository is organized into the following main components:

```
BDSN-FDC/
├── bayesian_networks                               # sample Bayesian networks saved in xdsl-files
├── data/                                           # datasets including sample fault cases
├── src/                                            # Main library for the scaling law analysis
│   ├── bdsn/                                       # source files of the BDSN-FDC software
│   ├── heuristic_parameter_identification/         # python files of the heuristic parameter identification
│   │   ├── *_bdsn_bn_based_on_fmeca_data.xdsl      # generated bayesian network based on FMECA data 
│   │   ├── heuristic_parameter_identification.py   # script for the heuristic parameter identification
│   │   └── inference_time_evaluation.py            # script for the runtime analysis of the inference   
│   ├── pemfp                                       # python files of the PEMFP algorithm
│   ├── bdsn_bayesian_network.py                    # python class of the bayesian network
│   └── bdsn_pgmpy_bayesian_network.py              # python class of the bayesian network using pgmpy
├── requirements.txt                                # Python dependencies
└── README.md                                       # This file
```


## Heuristic parameter identification approach using FMECA data

:file_folder: The file ´bayesian_networks/bayesian_network_based_on_fmeca_data.xdsl´ includes the definition of the Bayesian network that can be used with the [BayesFusion software](https://www.bayesfusion.com/) GeNIe Modeler and the pysmile wrapper.

:file_folder: The file ´data/fmeca_data.csv´ contains the integrated FMECA data used for the ontology-driven generation of the Bayesian network structure and the heuristic parameter identification approach using prior FMECA information.

:file_folder: The directory ´src/heuristic_parameter_identification´ contains the source code of the Bayesian network generation and the heuristic parameter identification approach using FMECA data (including the computational performance analysis) and for the computational performance analysis of the inference step. 

<!---
[comment]: # (### Citation)
[comment]: # (Please cite the following paper if you use this codebase:)
[comment]: # (```
@article{key,
  title        = {},
  author       = {Yannick Wilhelm and Peter Reimann and Wolfgang Gauchel and Bernhard Mitschang},
  year         = {},
  volume       = {},
  journal      = {},
  primaryclass = {},
  url          = {}
}
```)

[comment]: # (### Key Findings)

[comment]: # (- **Training**: asdf
- **Scalability**: asdf)
-->

## Penalized Expectation Maximization Algorithm with Fixed Parameters (PEMFP)
Content will be added after publication of the corresponding paper.

<!---[comment]: # (### Citation)
[comment]: # (Please cite the following paper if you use this codebase:)
[comment]: # (```
@article{key,
  title        = {},
  author       = {Yannick Wilhelm and Peter Reimann and Wolfgang Gauchel and Bernhard Mitschang},
  year         = {},
  volume       = {},
  journal      = {},
  primaryclass = {},
  url          = {}
}
```)-->

## Application BDSN-FDC
Content will be added after publication of the corresponding paper.

<!---
[comment]: # (### Citation)
[comment]: # (Please cite the following paper if you use this codebase:)
[comment]: # (```
@article{key,
  title        = {},
  author       = {Yannick Wilhelm and Peter Reimann and Wolfgang Gauchel and Bernhard Mitschang},
  year         = {},
  volume       = {},
  journal      = {},
  primaryclass = {},
  url          = {}
}
```)-->