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
  read.csv("/yourpath/Desktop/ML-Test-Smell-Detection-Online-Appendix/dataset/statisticalTestData/machineLearningAnalysis/cross/resultEagerTestCrossProject.csv",
           header = TRUE,
           sep = ",",
           dec = ".",
           row.names=NULL)
eagerTestInitial <- eagerTestInitial[complete.cases(eagerTestInitial),]

selected_models_ET <- subset(eagerTestInitial, ModelBalance %in% c("adaboost", "decisiontree", "multilayerperceptron", "naivebayes", "randomforest", "svm"))

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
        plottype = "mcb",ylab= "Likelihood MCC Eager Test Smell",xlab="",title=NULL,main=NULL,sub=NULL,cex.axis=5,cex.lab=1,cex.main=5,cex.sub=4)

#Crea il grafico a boxplot
ggplot(selected_models_ET, aes(x = ModelBalance, y = mcc, fill = ModelBalance)) +
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
  read.csv("/yourpath/Desktop/ML-Test-Smell-Detection-Online-Appendix/dataset/statisticalTestData/machineLearningAnalysis/cross/resultMysteryGuestCrossProject.csv",
           header = TRUE,
           sep = ",",
           dec = ".",
           row.names=NULL)
selected_models_MG <- subset(mysteryGuestInitial, ModelBalance %in% c("adaboost", "decisiontree", "multilayerperceptron", "naivebayes", "randomforest", "svm"))

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
        plottype = "mcb",ylab= "Likelihood MCC Mystery Guest Smell",xlab="",title=NULL,main=NULL,sub=NULL,cex.axis=5,cex.lab=1,cex.main=5,cex.sub=4)

#Crea il grafico a boxplot
ggplot(selected_models_MG, aes(x = ModelBalance, y = mcc, fill = ModelBalance)) +
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
  read.csv("/yourpath/Desktop/ML-Test-Smell-Detection-Online-Appendix/dataset/statisticalTestData/machineLearningAnalysis/cross/resultResourceOptimismCrossProject.csv",
           header = TRUE,
           sep = ",",
           dec = ".",
           row.names=NULL)
selected_models_RO <- subset(resourceOptimismInitial, ModelBalance %in% c("adaboost", "decisiontree", "multilayerperceptron", "naivebayes", "randomforest", "svm"))

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
        plottype = "mcb",ylab= "Likelihood MCC Resource Optimism Smell",xlab="",title=NULL,main=NULL,sub=NULL,cex.axis=5,cex.lab=1,cex.main=5,cex.sub=4)

#Crea il grafico a boxplot
ggplot(selected_models_RO, aes(x = ModelBalance, y = mcc, fill = ModelBalance)) +
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
  read.csv("/yourpath/Desktop/ML-Test-Smell-Detection-Online-Appendix/dataset/statisticalTestData/machineLearningAnalysis/cross/resultTestRedundancyCrossProject.csv",
           header = TRUE,
           sep = ",",
           dec = ".",
           row.names=NULL)
selected_models_TR <- subset(testRedundancyInitial, ModelBalance %in% c("adaboost", "decisiontree", "multilayerperceptron", "naivebayes", "randomforest", "svm"))


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
        plottype = "mcb",ylab= "Likelihood MCC Test Redundancy Smell",xlab="",title=NULL,main=NULL,sub=NULL,cex.axis=5,cex.lab=1,cex.main=5,cex.sub=4)

#Crea il grafico a boxplot
ggplot(selected_models_TR, aes(x = ModelBalance, y = mcc, fill = ModelBalance)) +
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

#eagerTestBalancing <- subset(eagerTestInitial, ModelBalance %in% c("adasyn_adaboost", "borderlinesmote_adaboost", "nearmissunder1_adaboost", "nearmissunder2_adaboost",
 #                                                                  "nearmissunder3_adaboost", "randomover_adaboost", "randomunder_adaboost", "smoteover_adaboost"))


yEagerTest <- matrix(c(
  eagerTestInitial[eagerTestInitial$ModelBalance == "adasyn_adaboost", ]$mcc,
  eagerTestInitial[eagerTestInitial$ModelBalance == "borderlinesmote_adaboost", ]$mcc,
  eagerTestInitial[eagerTestInitial$ModelBalance == "nearmissunder1_adaboost", ]$mcc,
  eagerTestInitial[eagerTestInitial$ModelBalance == "nearmissunder2_adaboost", ]$mcc,
  eagerTestInitial[eagerTestInitial$ModelBalance == "nearmissunder3_adaboost", ]$mcc,
  eagerTestInitial[eagerTestInitial$ModelBalance == "randomover_adaboost", ]$mcc,
  eagerTestInitial[eagerTestInitial$ModelBalance == "randomunder_adaboost", ]$mcc,
  eagerTestInitial[eagerTestInitial$ModelBalance == "smoteover_adaboost", ]$mcc,
  eagerTestInitial[eagerTestInitial$ModelBalance == "adaboost", ]$mcc),
  nrow=length(eagerTestInitial[eagerTestInitial$ModelBalance == "adasyn_adaboost", ]$mcc), ncol=9, 
  dimnames=list(1:length(eagerTestInitial[eagerTestInitial$ModelBalance == "adasyn_adaboost", ]$mcc),c(   "adasyn_AB",
                                                                                                          "borderlinesmote_AB",
                                                                                                          "nearmissunder1_AB",
                                                                                                          "nearmissunder2_AB",
                                                                                                          "nearmissunder3_AB",
                                                                                                          "randomover_AB",
                                                                                                          "randomunder_AB",
                                                                                                          "smoteover_AB",
                                                                                                          "Ada Boost")))

nemenyi(yEagerTest, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "MCC Eager Test Balanc. Cross",xlab="",title=NULL,main=NULL,sub=NULL,cex.axis=5,cex.lab=1,cex.main=5,cex.sub=4)



# Seleziona solo le righe con valori specifici di ModelBalance
dati_sel_EagerTest <- subset(eagerTestInitial, ModelBalance %in% c("adaboost","adasyn_adaboost", "borderlinesmote_adaboost", "nearmissunder1_adaboost", "nearmissunder2_adaboost",
                                                                   "nearmissunder3_adaboost", "randomover_adaboost", "randomunder_adaboost", "smoteover_adaboost"))

# Crea un vettore di colori casuali
colors <- sample(viridis::viridis_pal()(9), 9)

# Crea il grafico a boxplot con i colori casuali per Eager Test
ggplot(dati_sel_EagerTest, aes(x = ModelBalance, y = mcc, fill = ModelBalance)) +
  geom_boxplot() +
  labs(
    x = "",
    y = "MCC - Eager Test Cross") +
  scale_x_discrete(labels = c("Ada Boost","adasyn_AB", "borderlinesmote_AB", "nearmissunder1_AB",
                              "nearmissunder2_AB", "nearmissunder3_AB", "randomover_AB",
                              "randomunder_AB", "smoteover_AB")) + 
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "none"
  )

mysteryGuestBalancing <-
  read.csv("/yourpath/Desktop/ML-Test-Smell-Detection-Online-Appendix/dataset/statisticalTestData/machineLearningAnalysis/cross/resultMysteryGuestCrossProject.csv",
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
  mysteryGuestBalancing[mysteryGuestBalancing$ModelBalance == "randomforest", ]$mcc),
  nrow=length(mysteryGuestBalancing[mysteryGuestBalancing$ModelBalance == "adasyn_randomforest", ]$mcc), ncol=9, 
  dimnames=list(1:length(mysteryGuestBalancing[mysteryGuestBalancing$ModelBalance == "adasyn_randomforest", ]$mcc),c(   "adasyn_RF",
                                                                                                                        "borderlinesmote_RF",
                                                                                                                        "nearmissunder1_RF",
                                                                                                                        "nearmissunder2_RF",
                                                                                                                        "nearmissunder3_RF",
                                                                                                                        "randomover_RF",
                                                                                                                        "randomunder_RF",
                                                                                                                        "smoteover_RF",
                                                                                                                        "Random Forest")))


nemenyi(yMysteryGuest, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "MCC Mystery Guest Balanc. Cross",xlab="",title=NULL,main=NULL,sub=NULL,cex.axis=5,cex.lab=1,cex.main=5,cex.sub=4)


# Seleziona solo le righe con valori specifici di ModelBalance
dati_sel_MysteryGuest <- subset(mysteryGuestBalancing, ModelBalance %in% c("randomforest","adasyn_randomforest", "borderlinesmote_randomforest", "nearmissunder1_randomforest",
                                                                           "nearmissunder2_randomforest", "nearmissunder3_randomforest", "randomover_randomforest",
                                                                           "randomunder_randomforest", "smoteover_randomforest"))


# Crea il grafico a boxplot con i colori casuali per Mystery Guest
ggplot(dati_sel_MysteryGuest, aes(x = ModelBalance, y = mcc, fill = ModelBalance)) +
  geom_boxplot() +
  labs(
    x = "",
    y = "MCC - Mystery Guest Cross") +
  scale_x_discrete(labels = c("Random Forest","adasyn_RF", "borderlinesmote_RF", "nearmissunder1_RF",
                                                                  "nearmissunder2_RF", "nearmissunder3_RF", "randomover_RF",
                                                                  "randomunder_RF", "smoteover_RF")) + 
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "none"
  )



resourceOptimismBalancing <-
  read.csv("/yourpath/Desktop/ML-Test-Smell-Detection-Online-Appendix/dataset/statisticalTestData/machineLearningAnalysis/cross/resultResourceOptimismCrossProject.csv",
           header = TRUE,
           sep = ",",
           dec = ".",
           row.names=NULL)

yResourceOptimism <- matrix(c(
  resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "adasyn_svm", ]$mcc,
  resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "borderlinesmote_svm", ]$mcc,
  resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "nearmissunder1_svm", ]$mcc,
  resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "nearmissunder2_svm", ]$mcc,
  resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "nearmissunder3_svm", ]$mcc,
  resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "randomover_svm", ]$mcc,
  resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "randomunder_svm", ]$mcc,
  resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "smoteover_svm", ]$mcc,
  resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "svm", ]$mcc),
  nrow=length(resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "adasyn_svm", ]$mcc), ncol=9, 
  dimnames=list(1:length(resourceOptimismBalancing[resourceOptimismBalancing$ModelBalance == "adasyn_svm", ]$mcc),c(   "adasyn_SVM",
                                                                                                                       "borderlinesmote_SVM",
                                                                                                                       "nearmissunder1_SVM",
                                                                                                                       "nearmissunder2_SVM",
                                                                                                                       "nearmissunder3_SVM",
                                                                                                                       "randomover_SVM",
                                                                                                                       "randomunder_SVM",
                                                                                                                       "smoteover_SVM",
                                                                                                                       "SVM")))

nemenyi(yResourceOptimism, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "MCC Resource Opt. Balanc. Cross",xlab="",title=NULL,main=NULL,sub=NULL,cex.axis=5,cex.lab=1,cex.main=5,cex.sub=4)

# Seleziona solo le righe con valori specifici di ModelBalance
dati_sel_ResourceOptimism <- subset(resourceOptimismBalancing, ModelBalance %in% c("svm","adasyn_svm", "borderlinesmote_svm", "nearmissunder1_svm",
                                                                                   "nearmissunder2_svm", "nearmissunder3_svm", "randomover_svm",
                                                                                   "randomunder_svm", "smoteover_svm"))


# Crea il grafico a boxplot con i colori casuali per Mystery Guest
ggplot(dati_sel_ResourceOptimism, aes(x = ModelBalance, y = mcc, fill = ModelBalance)) +
  geom_boxplot() +
  labs(
    x = "",
    y = "MCC - Resource Optimism Cross")  + 
  scale_x_discrete(labels = c("SVM","adasyn_SVM", "borderlinesmote_SVM", "nearmissunder1_SVM",
                              "nearmissunder2_SVM", "nearmissunder3_SVM", "randomover_SVM",
                              "randomunder_SVM", "smoteover_SVM")) + 
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "none"
  )


testRedundancyBalancing <-
  read.csv("/yourpath/Desktop/ML-Test-Smell-Detection-Online-Appendix/dataset/statisticalTestData/machineLearningAnalysis/cross/resultTestRedundancyCrossProject.csv",
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
  dimnames=list(1:length(testRedundancyBalancing[testRedundancyBalancing$ModelBalance == "adasyn_naivebayes", ]$mcc),c(   "adasyn_NB",
                                                                                                                          "borderlinesmote_NB",
                                                                                                                          "nearmissunder1_NB",
                                                                                                                          "nearmissunder2_NB",
                                                                                                                          "nearmissunder3_NB",
                                                                                                                          "randomover_NB",
                                                                                                                          "randomunder_NB",
                                                                                                                          "smoteover_NB",
                                                                                                                          "Naive Bayes")))

nemenyi(yTestRedundancy, conf.level = 0.95, sort = TRUE,
        plottype = "mcb",ylab= "MCC Test Redund. Balanc. Cross",xlab="",title=NULL,main=NULL,sub=NULL,cex.axis=5,cex.lab=1,cex.main=5,cex.sub=4)


# Seleziona solo le righe con valori specifici di ModelBalance
dati_sel_testRedundancy <- subset(testRedundancyBalancing, ModelBalance %in% c("naivebayes","adasyn_naivebayes", "borderlinesmote_naivebayes", "nearmissunder1_naivebayes",
                                                                               "nearmissunder2_naivebayes", "nearmissunder3_naivebayes", "randomover_naivebayes",
                                                                               "randomunder_naivebayes", "smoteover_naivebayes"))


# Crea il grafico a boxplot con i colori casuali per Mystery Guest
ggplot(dati_sel_testRedundancy, aes(x = ModelBalance, y = mcc, fill = ModelBalance)) +
  geom_boxplot() +
  labs(
    x = "",
    y = "MCC - Test Redundancy Cross")  + 
  scale_x_discrete(labels = c("Naive Bayes","adasyn_NB", "borderlinesmote_NB", "nearmissunder1_NB",
                              "nearmissunder2_NB", "nearmissunder3_NB", "randomover_NB",
                              "randomunder_NB", "smoteover_NB")) + 
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "none"
  )


