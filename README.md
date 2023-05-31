# ML-Test-Smell-Detection-Online-Appendix
*Machine Learning-Based Test Smell Detection*

*Valeria Pontillo<sup>\*</sup>, Dario Amoroso d’Aragona<sup>†</sup>, Fabiano Pecorelli<sup>a</sup>,
Dario Di Nucci<sup>\*</sup>, Filomena Ferrucci<sup>\*</sup>*, Fabio Palomba<sup>\*</sup>

*<sup>*</sup>Software Engineering (SeSa) Lab — University of Salerno, Fisciano, Italy* </br>
*<sup>†</sup>Tampere University — Tampere, Finland* </br>
*<sup>a</sup>heronimus Academy of Data Science & Tilburg University, The Netherlands*

This repository contains the online appenddix of aforementioned work.

We take in account only the Test Smells detector tools that:
- Analyze Java project
- Uses metrics (rule-based)
- Uses Information Retrieval 


In this repository you can find:
- [Tools-Metrics Mapping](https://github.com/darioamorosodaragona-tuni/ML-Test-Smell-Detection-Online-Appendix/blob/main/Tools-Metrics%20Mapping.xlsx): an excel file showing for each test smell detector tool which metrics are used to detect the Test Smells investigated in our work:
    - Resource Optimism (RO)
    - Eager Test (ET)
    - Mystery Guest (MG)
    - Test Redundancy (TR)
- [projects.csv](https://github.com/darioamorosodaragona-tuni/ML-Test-Smell-Detection-Online-Appendix/blob/main/projects.csv): a csv file in which we report the url and SHA of the project that we will analyze..
- [projects_metrics.csv](https://github.com/darioamorosodaragona-tuni/ML-Test-Smell-Detection-Online-Appendix/blob/main/projects_metrics.csv): a csv file in which we report some information about the project that we will analyzwe i.e, n. of forks, stars, age.
- first_validation: folder with all the files regarding the first validation of the dataset performed by the two inspectors;
- dataset: folder with all the files regarding the various dataset used in our study. The most important are the folders "Eager Test", "Mystery Guest", "Resource Optimism", and "Test Redundancy" in which there are the dataset (splitted per project) with the various metrics computed for the machine learning-based approach. The dir "statisticalTestData" contains dataset for compute the nemenyi test in R;
- tool: folder with the two machine learning pipelines (one for the within-project validation and one for the cross-project validation). In ml_main, set your path correctly. In addition, In run_configuration, set the data to analyze if you want to apply the vif function and the other parameters related to the number of folds.
In addition, there are some R files useful to compute the Nemenyi Test.
- results: the various results splitted per research question. In RQ2 there are the results for all configuration and model created to validate our approach;
- externalValidation: in which are reported information about the external validation performed on Prolific, i.e., the sample used, the responses obtained, the user background and the responses discarded.


Contacts: </br>
vpontillo@unisa.it, dario.amorosodaragona@tuni.fi, f.pecorelli@jads.nl,
ddinucci@unisa.it, fferrucci@unisa.it, fpalomba@unisa.it





