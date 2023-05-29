library(MASS)
library(foreign)
library(nnet)
library(stargazer)
library(car)
library(tidyr)
library(effsize)
library(nortest)
library(ggplot2)

options(scipen=100000)

require(tsutils)
options(max.print = 10000)

#this is the Nemenyi test for EagerTest dataset
eagerTestInitial <-
  read.csv("/yourpath/ML-Test-Smell-Detection-Online-Appendix/dataset/statisticalTestData/detectorAnalysis/datasetForNemenyTest/eagerTestNemenyiTest.csv",
           header = TRUE,
           sep = ",",
           dec = ".",
           row.names=NULL)
#eagerTestInitial <- eagerTestInitial[complete.cases(eagerTestInitial),]

selected_models_ET <- subset(eagerTestInitial, ModelBalance %in% c("borderlinesmote_randomforest", "nearmissunder2_adaboost", "tsdetect_cross", "darts_cross"))

y <- matrix(c(
  eagerTestInitial[eagerTestInitial$ModelBalance == "borderlinesmote_randomforest", ]$mcc,
  eagerTestInitial[eagerTestInitial$ModelBalance == "nearmissunder2_adaboost", ]$mcc,
  eagerTestInitial[eagerTestInitial$ModelBalance == "tsdetect_cross", ]$mcc,
  eagerTestInitial[eagerTestInitial$ModelBalance == "darts_cross", ]$mcc),
  nrow=length(eagerTestInitial[eagerTestInitial$ModelBalance == "borderlinesmote_randomforest", ]$mcc), ncol=4, 
  dimnames=list(1:length(eagerTestInitial[eagerTestInitial$ModelBalance == "borderlinesmote_randomforest", ]$mcc),c("ML_within",
                                                                                                "ML_cross",
                                                                                                "tsdetect",
                                                                                                "darts"
                                                                                                )))

nemenyi(y, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "Likelihood MCC Eager Test Smell",xlab="",title=NULL,main=NULL,sub=NULL,cex.axis=5,cex.lab=1,cex.main=5,cex.sub=4)

#this is the Nemenyi test for MysteryGuest dataset
mysteryGuestInitial <-
  read.csv("/yourpath/ML-Test-Smell-Detection-Online-Appendix/dataset/statisticalTestData/detectorAnalysis/datasetForNemenyTest/mysteryGuestNemenyiTest.csv",
           header = TRUE,
           sep = ",",
           dec = ".",
           row.names=NULL)
#eagerTestInitial <- eagerTestInitial[complete.cases(eagerTestInitial),]

selected_models_MG <- subset(mysteryGuestInitial, ModelBalance %in% c("smoteover_randomforest", "nearmissunder1_randomforest", "tsdetect_within", "tsdetect_cross"))

x <- matrix(c(
  mysteryGuestInitial[mysteryGuestInitial$ModelBalance == "smoteover_randomforest", ]$mcc,
  mysteryGuestInitial[mysteryGuestInitial$ModelBalance == "nearmissunder1_randomforest", ]$mcc,
  mysteryGuestInitial[mysteryGuestInitial$ModelBalance == "tsdetect_cross", ]$mcc),
  nrow=length(mysteryGuestInitial[mysteryGuestInitial$ModelBalance == "smoteover_randomforest", ]$mcc), ncol=3, 
  dimnames=list(1:length(mysteryGuestInitial[mysteryGuestInitial$ModelBalance == "smoteover_randomforest", ]$mcc),c("ML_within",
                                                                                                               "ML_cross",
                                                                                                               "tsdetect"
                       
  )))

nemenyi(x, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "Likelihood MCC Mystery Guest Smell",xlab="",title=NULL,main=NULL,sub=NULL,cex.axis=5,cex.lab=1,cex.main=5,cex.sub=4)


#this is the Nemenyi test for ResourceOptimism dataset
resourceOptimismInitial <-
  read.csv("/yourpath/ML-Test-Smell-Detection-Online-Appendix/dataset/statisticalTestData/detectorAnalysis/datasetForNemenyTest/resourceOptimismNemenyiTest.csv",
           header = TRUE,
           sep = ",",
           dec = ".",
           row.names=NULL)
#eagerTestInitial <- eagerTestInitial[complete.cases(eagerTestInitial),]

selected_models_RO <- subset(resourceOptimismInitial, ModelBalance %in% c("borderlinesmote_randomforest", "randomunder_svm", "tsdetect_within", "tsdetect_cross"))

a <- matrix(c(
  resourceOptimismInitial[resourceOptimismInitial$ModelBalance == "borderlinesmote_randomforest", ]$mcc,
  resourceOptimismInitial[resourceOptimismInitial$ModelBalance == "randomunder_svm", ]$mcc,
  resourceOptimismInitial[resourceOptimismInitial$ModelBalance == "tsdetect_cross", ]$mcc),
  nrow=length(resourceOptimismInitial[resourceOptimismInitial$ModelBalance == "borderlinesmote_randomforest", ]$mcc), ncol=3, 
  dimnames=list(1:length(resourceOptimismInitial[resourceOptimismInitial$ModelBalance == "borderlinesmote_randomforest", ]$mcc),c("ML_within",
                                                                                                                    "ML_cross",
                                                                                                                    "tsdetect"
                                                                                                                    
  )))

nemenyi(a, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "Likelihood MCC Resource Optimism Smell",xlab="",title=NULL,main=NULL,sub=NULL,cex.axis=5,cex.lab=1,cex.main=5,cex.sub=4)



#this is the Nemenyi test for TestRedundancy dataset
testRedundancyInitial <-
  read.csv("/yourpath/ML-Test-Smell-Detection-Online-Appendix/dataset/statisticalTestData/detectorAnalysis/datasetForNemenyTest/testRedundancyNemenyiTest.csv",
           header = TRUE,
           sep = ",",
           dec = ".",
           row.names=NULL)
#eagerTestInitial <- eagerTestInitial[complete.cases(eagerTestInitial),]

selected_models_TR <- subset(testRedundancyInitial, ModelBalance %in% c("smoteover_naivebayes_within", "smoteover_naivebayes", "teredetect_within", "teredetect_cross"))

c <- matrix(c(
  testRedundancyInitial[testRedundancyInitial$ModelBalance == "smoteover_naivebayes_within", ]$mcc,
  testRedundancyInitial[testRedundancyInitial$ModelBalance == "smoteover_naivebayes", ]$mcc,
  testRedundancyInitial[testRedundancyInitial$ModelBalance == "teredetect_cross", ]$mcc),
  nrow=length(testRedundancyInitial[testRedundancyInitial$ModelBalance == "smoteover_naivebayes_within", ]$mcc), ncol=3, 
  dimnames=list(1:length(testRedundancyInitial[testRedundancyInitial$ModelBalance == "smoteover_naivebayes_within", ]$mcc),c("ML_within",
                                                                                                                                  "ML_cross",
                                                                                                                                  "teredetect"
                                                                                                                                  
  )))

nemenyi(c, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "Likelihood MCC Test Redundancy Smell",xlab="",title=NULL,main=NULL,sub=NULL,cex.axis=5,cex.lab=1,cex.main=5,cex.sub=4)

