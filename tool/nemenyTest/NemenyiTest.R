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
  read.csv("/yourpath/ML-Test-Smell-Detection-Online-Appendix/dataset/statisticalTestData/machineLearningAnalysis/within/initialModel/modelForStatisticalTest/eagerTest.csv",
           header = TRUE,
           sep = ",",
           dec = ".",
           row.names=NULL)
eagerTestInitial <- eagerTestInitial[complete.cases(eagerTestInitial),]

y <- matrix(c(
  eagerTestInitial[eagerTestInitial$ModelBalance == "adaboost", ]$mcc,
  eagerTestInitial[eagerTestInitial$ModelBalance == "decisiontree", ]$mcc,
  eagerTestInitial[eagerTestInitial$ModelBalance == "multilayerperceptron", ]$mcc,
  eagerTestInitial[eagerTestInitial$ModelBalance == "naivebayes", ]$mcc,
  eagerTestInitial[eagerTestInitial$ModelBalance == "randomforest", ]$mcc,
  eagerTestInitial[eagerTestInitial$ModelBalance == "svm", ]$mcc),
  nrow=length(eagerTestInitial[eagerTestInitial$ModelBalance == "adaboost", ]$mcc), ncol=6, 
  dimnames=list(1:length(eagerTestInitial[eagerTestInitial$ModelBalance == "adaboost", ]$mcc),c("adaboost",
                                                                                     "decisiontree",
                                                                                     "multilayerperceptron",
                                                                                     "naivebayes",
                                                                                     "randomforest",
                                                                                     "svm")))

nemenyi(y, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "Likelihood MCC Eager Test",xlab="",title=NULL,main=NULL,sub=NULL,cex.axis=5,cex.lab=1,cex.main=5,cex.sub=4)

#Crea il grafico a boxplot
ggplot(eagerTestInitial, aes(x = ModelBalance, y = mcc, fill = ModelBalance)) +
  geom_boxplot() +
  labs(
       x = "",
       y = "MCC - Eager Test") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "none"
        ) +
  scale_fill_manual(values = c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00"))

#this is the Nemenyi test for MysterGuest dataset
mysteryGuestInitial <-
  read.csv("/Yourpath/ML-Test-Smell-Detection-Online-Appendix/dataset/statisticalTestData/machineLearningAnalysis/within/initialModel/modelForStatisticalTest/mysteryGuest.csv",
           header = TRUE,
           sep = ",",
           dec = ".",
           row.names=NULL)

x <- matrix(c(
  mysteryGuestInitial[mysteryGuestInitial$ModelBalance == "adaboost", ]$mcc,
  mysteryGuestInitial[mysteryGuestInitial$ModelBalance == "decisiontree", ]$mcc,
  mysteryGuestInitial[mysteryGuestInitial$ModelBalance == "multilayerperceptron", ]$mcc,
  mysteryGuestInitial[mysteryGuestInitial$ModelBalance == "naivebayes", ]$mcc,
  mysteryGuestInitial[mysteryGuestInitial$ModelBalance == "randomforest", ]$mcc,
  mysteryGuestInitial[mysteryGuestInitial$ModelBalance == "svm", ]$mcc),
  nrow=length(mysteryGuestInitial[mysteryGuestInitial$ModelBalance == "adaboost", ]$mcc), ncol=6, 
  dimnames=list(1:length(mysteryGuestInitial[mysteryGuestInitial$ModelBalance == "adaboost", ]$mcc),c(   "adaboost",
                                                                                                   "decisiontree",
                                                                                                   "multilayerperceptron",
                                                                                                   "naivebayes",
                                                                                                   "randomforest",
                                                                                                   "svm")))

nemenyi(x, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "Likelihood MCC Mystery Guest",xlab="",title=NULL,main=NULL,sub=NULL,cex.axis=5,cex.lab=1,cex.main=5,cex.sub=4)

#Crea il grafico a boxplot
ggplot(mysteryGuestInitial, aes(x = ModelBalance, y = mcc, fill = ModelBalance)) +
  geom_boxplot() +
  labs(
    x = "",
    y = "MCC - Mystery Guest") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "none"
  ) +
  scale_fill_manual(values = c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00"))

#this is the Nemenyi test for Resource Optimism dataset
resourceOptimismInitial <-
  read.csv("/YourPath/ML-Test-Smell-Detection-Online-Appendix/dataset/statisticalTestData/machineLearningAnalysis/within/initialModel/modelForStatisticalTest/resourceOptimism.csv",
           header = TRUE,
           sep = ",",
           dec = ".",
           row.names=NULL)

z <- matrix(c(
  resourceOptimismInitial[resourceOptimismInitial$ModelBalance == "adaboost", ]$mcc,
  resourceOptimismInitial[resourceOptimismInitial$ModelBalance == "decisiontree", ]$mcc,
  resourceOptimismInitial[resourceOptimismInitial$ModelBalance == "multilayerperceptron", ]$mcc,
  resourceOptimismInitial[resourceOptimismInitial$ModelBalance == "naivebayes", ]$mcc,
  resourceOptimismInitial[resourceOptimismInitial$ModelBalance == "randomforest", ]$mcc,
  resourceOptimismInitial[resourceOptimismInitial$ModelBalance == "svm", ]$mcc),
  nrow=length(resourceOptimismInitial[resourceOptimismInitial$ModelBalance == "adaboost", ]$mcc), ncol=6, 
  dimnames=list(1:length(resourceOptimismInitial[resourceOptimismInitial$ModelBalance == "adaboost", ]$mcc),c(   "adaboost",
                                                                                                         "decisiontree",
                                                                                                         "multilayerperceptron",
                                                                                                         "naivebayes",
                                                                                                         "randomforest",
                                                                                                         "svm")))

nemenyi(z, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "Likelihood MCC Resource Optimism",xlab="",title=NULL,main=NULL,sub=NULL,cex.axis=5,cex.lab=1,cex.main=5,cex.sub=4)

#Crea il grafico a boxplot
ggplot(resourceOptimismInitial, aes(x = ModelBalance, y = mcc, fill = ModelBalance)) +
  geom_boxplot() +
  labs(
    y = "MCC - Resource Optimism",
    x = "") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "none"
  ) +
  scale_fill_manual(values = c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00"))



#this is the Nemenyi test for Test Redundancy dataset
testRedundancyInitial <-
  read.csv("/YourPAth/ML-Test-Smell-Detection-Online-Appendix/dataset/statisticalTestData/machineLearningAnalysis/within/initialModel/modelForStatisticalTest/testRedundancy.csv",
           header = TRUE,
           sep = ",",
           dec = ".",
           row.names=NULL)

w <- matrix(c(
  testRedundancyInitial[testRedundancyInitial$ModelBalance == "adaboost", ]$mcc,
  testRedundancyInitial[testRedundancyInitial$ModelBalance == "decisiontree", ]$mcc,
  testRedundancyInitial[testRedundancyInitial$ModelBalance == "multilayerperceptron", ]$mcc,
  testRedundancyInitial[testRedundancyInitial$ModelBalance == "naivebayes", ]$mcc,
  testRedundancyInitial[testRedundancyInitial$ModelBalance == "randomforest", ]$mcc,
  testRedundancyInitial[testRedundancyInitial$ModelBalance == "svm", ]$mcc),
  nrow=length(testRedundancyInitial[testRedundancyInitial$ModelBalance == "adaboost", ]$mcc), ncol=6, 
  dimnames=list(1:length(testRedundancyInitial[testRedundancyInitial$ModelBalance == "adaboost", ]$mcc),c(   "adaboost",
                                                                                                                 "decisiontree",
                                                                                                                 "multilayerperceptron",
                                                                                                                 "naivebayes",
                                                                                                                 "randomforest",
                                                                                                                 "svm")))

nemenyi(w, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "Likelihood MCC Test Redundancy",xlab="",title=NULL,main=NULL,sub=NULL,cex.axis=5,cex.lab=1,cex.main=5,cex.sub=4)

#Crea il grafico a boxplot
ggplot(testRedundancyInitial, aes(x = ModelBalance, y = mcc, fill = ModelBalance)) +
  geom_boxplot() +
  labs(
    x = "",
    y = "MCC - Test Redundancy") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "none"
  ) +
  scale_fill_manual(values = c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00"))

statistiche <- tapply(testRedundancyInitial$mcc, testRedundancyInitial$ModelBalance, function(x) c(min = min(x), quantile(x, probs = c(0.25, 0.5, 0.75)), max = max(x)))

# Stampa dei risultati
print(statistiche)


#_______________________________Comparison between the various balancing techiniques_________________________


yEagerTest <- matrix(c(
  eagerTestBalancing[eagerTestBalancing$ModelBalance == "adasyn_randomforest", ]$mcc,
  eagerTestBalancing[eagerTestBalancing$ModelBalance == "borderlinesmote_randomforest", ]$mcc,
  eagerTestBalancing[eagerTestBalancing$ModelBalance == "nearmissunder1_randomforest", ]$mcc,
  eagerTestBalancing[eagerTestBalancing$ModelBalance == "nearmissunder2_randomforest", ]$mcc,
  eagerTestBalancing[eagerTestBalancing$ModelBalance == "nearmissunder3_randomforest", ]$mcc,
  eagerTestBalancing[eagerTestBalancing$ModelBalance == "randomover_randomforest", ]$mcc,
  eagerTestBalancing[eagerTestBalancing$ModelBalance == "randomunder_randomforest", ]$mcc,
  eagerTestBalancing[eagerTestBalancing$ModelBalance == "smoteover_randomforest", ]$mcc,
  eagerTestBalancing[eagerTestBalancing$ModelBalance == "randomforest", ]$mcc),
  nrow=length(eagerTestBalancing[eagerTestBalancing$ModelBalance == "adasyn_randomforest", ]$mcc), ncol=9, 
  dimnames=list(1:length(eagerTestBalancing[eagerTestBalancing$ModelBalance == "adasyn_randomforest", ]$mcc),c(   "adasyin_RF",
                                                                                     "borderlinesmote_RF",
                                                                                     "nearmissunder1_RF",
                                                                                     "nearmissunder2_RF",
                                                                                     "nearmissunder3_RF",
                                                                                     "randomover_RF",
                                                                                     "randomunder_RF",
                                                                                     "smoteover_RF",
                                                                                     "Random Forest")))

nemenyi(yEagerTest, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "Likelihood MCC Eager Test Balanc.",xlab="",title=NULL,main=NULL,sub=NULL,cex.axis=5,cex.lab=1,cex.main=5,cex.sub=4)



# Seleziona solo le righe con valori specifici di ModelBalance
dati_sel_EagerTest <- subset(eagerTestBalancing, ModelBalance %in% c("randomforest","adasyn_randomforest", "borderlinesmote_randomforest", "nearmissunder1_randomforest",
                                                           "nearmissunder2_randomforest", "nearmissunder3_randomforest", "randomover_randomforest",
                                                           "randomunder_randomforest", "smoteover_randomforest"))

# Crea un vettore di colori casuali
colors <- sample(viridis::viridis_pal()(9), 9)
print(colors)

# Crea il grafico a boxplot con i colori casuali per Eager Test
ggplot(dati_sel_EagerTest, aes(x = ModelBalance, y = mcc, fill = ModelBalance)) +
  geom_boxplot() +
  labs(
    x = "",
    y = "MCC - Eager Test") +
  scale_x_discrete(labels = c("Random Forest","adasyn_RF", "borderlinesmote_RF", "nearmissunder1_RF",
                              "nearmissunder2_RF", "nearmissunder3_RF", "randomover_RF",
                              "randomunder_RF", "smoteover_RF")) + 
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "none"
  )

mysteryGuestBalancing <-
  read.csv("/yourPath/ML-Test-Smell-Detection-Online-Appendix/dataset/statisticalTestData/machineLearningAnalysis/within/allModels/resultMysteryGuest.csv",
           header = TRUE,
           sep = ",",
           dec = ".",
           row.names=NULL)


yMysteryGuest <- matrix(c(
  mysteryGuestBalancing[mysteryGuestBalancing$ModelBalance == "adasyn_randomforest", ]$mcc,
  mysteryGuestBalancing[mysteryGuestBalancing$ModelBalance == "borderlinesmote_randomforest", ]$mcc,
  mysteryGuestBalancing[mysteryGuestBalancing$ModelBalance == "nearmissunder1_randomforest", ]$mcc,
  mysteryGuestBalancing[mysteryGuestBalancing$ModelBalance == "nearmissunder2_randomforest", ]$mcc,
  mysteryGuestBalancing[mysteryGuestBalancing$ModelBalance == "nearmissunder3_randomforest", ]$mcc,
  mysteryGuestBalancing[mysteryGuestBalancing$ModelBalance == "randomover_randomforest", ]$mcc,
  mysteryGuestBalancing[mysteryGuestBalancing$ModelBalance == "randomunder_randomforest", ]$mcc,
  mysteryGuestBalancing[mysteryGuestBalancing$ModelBalance == "smoteover_randomforest", ]$mcc,
  mysteryGuestBalancing[mysteryGuestBalancing$ModelBalance == "randomforest", ]$mc),
  nrow=length(mysteryGuestBalancing[mysteryGuestBalancing$ModelBalance == "adasyn_randomforest", ]$mcc), ncol=9, 
  dimnames=list(1:length(mysteryGuestBalancing[mysteryGuestBalancing$ModelBalance == "adasyn_randomforest", ]$mcc),c(   "adasyin_RF",
                                                                                                                         "borderlinesmote_RF",
                                                                                                                         "nearmissunder1_RF",
                                                                                                                         "nearmissunder2_RF",
                                                                                                                         "nearmissunder3_RF",
                                                                                                                         "randomover_RF",
                                                                                                                         "randomunder_RF",
                                                                                                                         "smoteover_RF",
                                                                                                                          "Random Forest")))

nemenyi(yMysteryGuest, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "Likelihood MCC Mystery Guest Balanc.",xlab="",title=NULL,main=NULL,sub=NULL,cex.axis=5,cex.lab=1,cex.main=5,cex.sub=4)


# Seleziona solo le righe con valori specifici di ModelBalance
dati_sel_MysteryGuest <- subset(mysteryGuestBalancing, ModelBalance %in% c("randomforest","adasyn_randomforest", "borderlinesmote_randomforest", "nearmissunder1_randomforest",
                                                                           "nearmissunder2_randomforest", "nearmissunder3_randomforest", "randomover_randomforest",
                                                                           "randomunder_randomforest", "smoteover_randomforest"))


# Crea il grafico a boxplot con i colori casuali per Mystery Guest
ggplot(dati_sel_MysteryGuest, aes(x = ModelBalance, y = mcc, fill = ModelBalance)) +
  geom_boxplot() +
  labs(
    x = "",
    y = "MCC - Mystery Guest") +
  scale_x_discrete(labels = c("Random Forest","adasyn_RF", "borderlinesmote_RF", "nearmissunder1_RF",
                              "nearmissunder2_RF", "nearmissunder3_RF", "randomover_RF",
                              "randomunder_RF", "smoteover_RF")) + 
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "none"
  )



resourceOptimismBalancing <-
  read.csv("/yourPath/ML-Test-Smell-Detection-Online-Appendix/dataset/statisticalTestData/machineLearningAnalysis/within/allModels/resultResourceOptimism.csv",
           header = TRUE,
           sep = ",",
           dec = ".",
           row.names=NULL)

yResourceOptimism <- matrix(c(
  resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "adasyn_randomforest", ]$mcc,
  resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "borderlinesmote_randomforest", ]$mcc,
  resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "nearmissunder1_randomforest", ]$mcc,
  resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "nearmissunder2_randomforest", ]$mcc,
  resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "nearmissunder3_randomforest", ]$mcc,
  resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "randomover_randomforest", ]$mcc,
  resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "randomunder_randomforest", ]$mcc,
  resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "smoteover_randomforest", ]$mcc,
  resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "randomforest", ]$mcc),
  nrow=length(resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "adasyn_randomforest", ]$mcc), ncol=9, 
  dimnames=list(1:length(resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "adasyn_randomforest", ]$mcc),c(   "adasyin_RF",
                                                                                                                                 "borderlinesmote_RF",
                                                                                                                                 "nearmissunder1_RF",
                                                                                                                                 "nearmissunder2_RF",
                                                                                                                                 "nearmissunder3_RF",
                                                                                                                                 "randomover_RF",
                                                                                                                                 "randomunder_RF",
                                                                                                                                 "smoteover_RF",
                                                                                                                                "Random Forest")))

nemenyi(yResourceOptimism, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "Likelihood MCC Resource Opt. Balanc.",xlab="",title=NULL,main=NULL,sub=NULL,cex.axis=5,cex.lab=1,cex.main=5,cex.sub=4)

# Seleziona solo le righe con valori specifici di ModelBalance
dati_sel_ResourceOptimism <- subset(resourceOptimismBalancing, ModelBalance %in% c("randomforest","adasyn_randomforest", "borderlinesmote_randomforest", "nearmissunder1_randomforest",
                                                                           "nearmissunder2_randomforest", "nearmissunder3_randomforest", "randomover_randomforest",
                                                                           "randomunder_randomforest", "smoteover_randomforest"))


# Crea il grafico a boxplot con i colori casuali per Mystery Guest
ggplot(dati_sel_ResourceOptimism, aes(x = ModelBalance, y = mcc, fill = ModelBalance)) +
  geom_boxplot() +
  labs(
    x = "",
    y = "MCC - Resource Optimism") +
  scale_x_discrete(labels = c("Random Forest","adasyn_RF", "borderlinesmote_RF", "nearmissunder1_RF",
                              "nearmissunder2_RF", "nearmissunder3_RF", "randomover_RF",
                              "randomunder_RF", "smoteover_RF")) + 
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "none"
  )


testRedundancyBalancing <-
  read.csv("/yourPath/ML-Test-Smell-Detection-Online-Appendix/dataset/statisticalTestData/machineLearningAnalysis/within/allModels/resultTestRedundancy.csv",
           header = TRUE,
           sep = ",",
           dec = ".",
           row.names=NULL)

yTestRedundancy <- matrix(c(
  testRedundancyBalancing[testRedundancyBalancing$ModelBalance == "adasyn_naivebayes", ]$mcc,
  testRedundancyBalancing[testRedundancyBalancing$ModelBalance == "borderlinesmote_naivebayes", ]$mcc,
  testRedundancyBalancing[testRedundancyBalancing$ModelBalance == "nearmissunder1_naivebayes", ]$mcc,
  testRedundancyBalancing[testRedundancyBalancing$ModelBalance == "nearmissunder2_naivebayes", ]$mcc,
  testRedundancyBalancing[testRedundancyBalancing$ModelBalance == "nearmissunder3_naivebayes", ]$mcc,
  testRedundancyBalancing[testRedundancyBalancing$ModelBalance == "randomover_naivebayes", ]$mcc,
  testRedundancyBalancing[testRedundancyBalancing$ModelBalance == "randomunder_naivebayes", ]$mcc,
  testRedundancyBalancing[testRedundancyBalancing$ModelBalance == "smoteover_naivebayes", ]$mcc,
  testRedundancyBalancing[testRedundancyBalancing$ModelBalance == "naivebayes", ]$mcc),
  nrow=length(testRedundancyBalancing[testRedundancyBalancing$ModelBalance == "adasyn_naivebayes", ]$mcc), ncol=9, 
  dimnames=list(1:length(testRedundancyBalancing[testRedundancyBalancing$ModelBalance == "adasyn_naivebayes", ]$mcc),c("adasyin_NB",
                                                                                                                              "borderlinesmote_NB",
                                                                                                                              "nearmissunder1_NB",
                                                                                                                              "nearmissunder2_NB",
                                                                                                                              "nearmissunder3_NB",
                                                                                                                              "randomover_NB",
                                                                                                                              "randomunder_NB",
                                                                                                                              "smoteover_NB",
                                                                                                                       "Naive Bayes"
                                                                                                                             )))

nemenyi(yTestRedundancy, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "Likelihood MCC Test Red. Balanc.",xlab="",title=NULL,main=NULL,sub=NULL,cex.axis=5,cex.lab=1,cex.main=5,cex.sub=4)


# Seleziona solo le righe con valori specifici di ModelBalance
dati_sel_testRedundancy <- subset(testRedundancyBalancing, ModelBalance %in% c("naivebayes","adasyn_naivebayes", "borderlinesmote_naivebayes", "nearmissunder1_naivebayes",
                                                                                   "nearmissunder2_naivebayes", "nearmissunder3_naivebayes", "randomover_naivebayes",
                                                                                   "randomunder_naivebayes", "smoteover_naivebayes"))


# Crea il grafico a boxplot con i colori casuali per Mystery Guest
ggplot(dati_sel_testRedundancy, aes(x = ModelBalance, y = mcc, fill = ModelBalance)) +
  geom_boxplot() +
  
  labs(
    x = "",
    y = "MCC - Test Redundancy") + 
  scale_x_discrete(labels = c("Naive Bayes","adasyn_NB", "borderlinesmote_NB", "nearmissunder1_NB",
                              "nearmissunder2_NB", "nearmissunder3_NB", "randomover_NB",
                              "randomunder_NB", "smoteover_NB")) + 
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "none"
        
  )


