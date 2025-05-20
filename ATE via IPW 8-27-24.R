#install.packages('emmeans', repos='https://cran.r-project.org/')
#install.packages('emmeans')

library(model4you)
library(dplyr)
library(sqldf)
library(tictoc)
library(caret)
library(emmeans)
library(tableone)
library(survey)
library(ggplot2)

setwd('//v06.med.va.gov/DUR/HSRD/HTE_KOA_IRB_1725969/8_Statistician (sensitive data)/Statistician/Data/')

df = read.csv('KOAHTE_dataset.csv') %>%
  mutate(treat = as.factor(ifelse(SKOAArm==0 & is.na(GroupPTArm), "Control", 
                            ifelse(SKOAArm==1 & is.na(GroupPTArm), "SKOA",
                                   ifelse(GroupPTArm==0 & is.na(SKOAArm), "IndividualPT", 
                                          ifelse(GroupPTArm==1 & is.na(SKOAArm),"GroupPT",NA)))))) %>%
  mutate(WomacTotal_Delta = use_WomacTotal_Post-use_WomacTotal_Pre) %>%
  select(c('STUDYID','YearsWithArthritis_Pre','SelfRatedHealth_Pre','BMI_Pre','Age_Pre',
           'Education_Pre','WorkStatus_Pre','Gender_Pre','Ethnicity_Pre','SelfEfficacyExerciseTotal_Pre',
           'White1Nonwhite0KOAHTE_Pre','OpioidMed_Pre','NonOpioidPainMed_Pre',
           'AssistiveDevBrace_Pre','AssistiveDevCaneStick_Pre','StrengthPASECHAMPS_Pre',
           'ComorbidityHeartDisease_Pre','ComorbidityHighBloodPressure_Pre',
           'ComorbidityDiabetes_Pre','ComorbidityDepression_Pre',
           'ComorbidityBackPain_Pre',
           'TotalJointsArthritis_Pre','LowerBodyJointsArthritis_Pre',
           'LowBackArthritis_Pre','Bilateral1Unilateral0KOA_Pre','use_WomacTotal_Pre',
           'treat','WomacTotal_Delta'))

df_complete = df[complete.cases(df),]

# setting up cross-validation
# elastic net model - tuning of hyperparameters performed by caret
cv_5_bin = trainControl(method = 'repeatedcv', repeats = 10, number = 5, 
                        classProbs = TRUE)

tic('model training')
model = train(treat ~ . -WomacTotal_Delta -STUDYID,
              data = df_complete,
              method = 'xgbTree',
              trControl = cv_5_bin,
              metric = "ROC",
              na.action=na.exclude)
toc() #took 22.5 minutes

saveRDS(model, 'XGBoost Propensity Model 8-27-24.RData')
model = readRDS('XGBoost Propensity Model 8-27-24.RData')

get_best_result = function(caret_fit){
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
  best_result = caret_fit$result[best, ]
  rownames(best_result) = NULL
  best_result
}

get_best_result(model)

summary(model$results)

model2 = train(treat ~ . -WomacTotal_Delta -STUDYID,
               data = df_complete,
               method = 'xgbTree',
               trControl = trainControl(
                 method = "none"),
               tuneGrid = data.frame(eta = 0.3, max_depth=1, gamma=0,
                                     colsample_bytree=0.6, min_child_weight=1,
                                     subsample=1, nrounds=50),
               metric = "Kappa",
               na.action=na.exclude)

summary(model2)

pred_outcome = predict(model2, newdata = df_complete, type="raw") 
pred_prob = predict(model2, newdata = df_complete, type="prob") 

table(pred_outcome)
table(pred_outcome, df_complete$treat)

cm = caret::confusionMatrix(table(pred_outcome, df_complete$treat))

props = prop.table(table(df_complete$treat))

#create IPW
df_complete$IPW = ifelse(df_complete$treat=="Control", props[1]/pred_prob$Control,
                         ifelse(df_complete$treat=="GroupPT", props[2]/pred_prob$GroupPT,
                                ifelse(df_complete$treat=="IndividualPT", props[3]/pred_prob$IndividualPT,
                                       props[4]/pred_prob$SKOA)))

summary(df_complete$IPW)

#examine covariate balance
design = svydesign(ids=~1, data=df_complete, weights=~IPW)

tab3 = svyCreateTableOne(vars=colnames(df_complete %>% select(-c(WomacTotal_Delta,STUDYID))),
                      data=design,
                      strata="treat")

print(tab3, smd=T)

#examine overlap
ggplot(data=df_complete, aes(x=IPW, fill=treat)) +
  geom_histogram(position="identity", alpha=0.1, bins=30) +
  theme_minimal()

####################
# Marginal Effects #
####################

lm = lm(WomacTotal_Delta ~ treat, data=df_complete, weights=IPW)

posthoc = emmeans(lm, data=df_complete, ~treat) %>% data.frame %>%
  mutate(treat2 = c((paste('Control','(n=107)',sep='\n')),
                    (paste('Group PT','(n=153)',sep='\n')),
                    (paste('Individual PT','(n=147)',sep='\n')),
                    (paste('STEP KOA','(n=214)',sep='\n'))))

ggplot(data=posthoc, aes(y=emmean, x=treat2, ymin=lower.CL, ymax=upper.CL)) +
  geom_point(stat="identity", size=3) +
  geom_errorbar(width=0.1) + 
  scale_y_continuous(breaks=seq(-10,6,2)) +
  xlab("") +
  ylab("WOMAC Change Score") +
  theme_classic()

ggsave("ATE figure 1-1-25.jpeg", width=7.5, height=3.5, dpi=600)


#weighted ANOVA

summary(df_complete$IPW)

anova = aov(WomacTotal_Delta~treat, data=df_complete, weights=IPW)
summary(anova)

library(Hmisc)
library(rstatix) 


for (trx in c('Control','GroupPT','IndividualPT','SKOA')){
  print(trx)
  
  temp = df_complete[df_complete$treat==trx,]
  
  print(paste0('mean (SD) = ',sprintf('%.2f',wtd.mean(temp$WomacTotal_Delta, temp$IPW, normwt=FALSE)), ' (',
               sprintf('%.2f',sqrt(wtd.var(temp$WomacTotal_Delta, temp$IPW, normwt=FALSE))), ')'))
}

TukeyHSD(anova)

