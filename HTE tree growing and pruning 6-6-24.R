library(model4you)
library(dplyr)
library(sqldf)
library(tictoc)
library(caret)
library(emmeans)

setwd('//v06.med.va.gov/DUR/HSRD/HTE_KOA_IRB_1725969/8_Statistician (sensitive data)/Statistician/Data/')

df = read.csv('KOAHTE_dataset.csv') %>%
  mutate(treat = as.factor(ifelse(SKOAArm==0 & is.na(GroupPTArm), "Control", 
                            ifelse(SKOAArm==1 & is.na(GroupPTArm), "SKOA",
                                   ifelse(GroupPTArm==0 & is.na(SKOAArm), "IndividualPT", 
                                          ifelse(GroupPTArm==1 & is.na(SKOAArm),"GroupPT",NA)))))) %>%
  mutate(WomacTotal_Delta = use_WomacTotal_Post-use_WomacTotal_Pre) %>%
  select(c('STUDYID','YearsWithArthritis_Pre','SelfRatedHealth_Pre','BMI_Pre','Age_Pre',
           'Education_Pre','WorkStatus_Pre','Gender_Pre','Ethnicity_Pre',
           'White1Nonwhite0KOAHTE_Pre','OpioidMed_Pre','NonOpioidPainMed_Pre',
           'AssistiveDevBrace_Pre','AssistiveDevCaneStick_Pre','StrengthPASECHAMPS_Pre',
           'ComorbidityHeartDisease_Pre','ComorbidityHighBloodPressure_Pre',
           'ComorbidityDiabetes_Pre','ComorbidityDepression_Pre',
           'ComorbidityBackPain_Pre','SelfEfficacyExerciseTotal_Pre',
           'TotalJointsArthritis_Pre','LowerBodyJointsArthritis_Pre',
           'LowBackArthritis_Pre','Bilateral1Unilateral0KOA_Pre','use_WomacTotal_Pre',
           'treat','WomacTotal_Delta'))

df_complete = df[complete.cases(df),]

table(df_complete$treat)

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

cm = caret::confusionMatrix(table(pred_outcome, df_complete$treat))

props = prop.table(table(df_complete$treat))

#create IPW
df_complete$IPW = ifelse(df_complete$treat=="Control", props[1]/pred_prob$Control,
                         ifelse(df_complete$treat=="GroupPT", props[2]/pred_prob$GroupPT,
                                ifelse(df_complete$treat=="IndividualPT", props[3]/pred_prob$IndividualPT,
                                       props[4]/pred_prob$SKOA)))


### need to manually program cross-validation, probability 7-fold cross validation,
### varying depth and p-value of splits

## splitting the data ##

training_IDs = read.csv('Python training set.csv')

train = df_complete %>% select(-STUDYID)

summary(train$IPW)

## set up grid search  ##

# store results here
results = data.frame(matrix(ncol=102,nrow=0))
colnames(results) = c('depth','p')

tic('loop')
# loop over repetitions
for (reps in seq(0:9)){
  
  #split train into 5 folds
  index = sample(1:nrow(train), .2*nrow(train)); fold1=train[index,]; remainder = train[-index,]
  index = sample(1:nrow(remainder), .2*nrow(train)); fold2=remainder[index,]; remainder = remainder[-index,]
  index = sample(1:nrow(remainder), .2*nrow(train)); fold3=remainder[index,]; remainder = remainder[-index,]
  index = sample(1:nrow(remainder), .2*nrow(train)); fold4=remainder[index,]; fold5 = remainder[-index,]
  
  i=1
  
  # loop over these values for depth
  for (depth in seq(2,10,1)){
    
    #loop over these values for p-value
    for (p in seq(.05,.9,.01)){
      
      results[i,1] = depth
      results[i,2] = p
      
      #loop over these folds
      for (k in seq(1,5,1)){
        
        #      depth=5
        #      p=.5
        #      k=2
        
        #creating sub training and test sets for model training
        if (k==1){
          train_sub=rbind(fold1,fold2,fold3,fold4); test_sub=fold5
        } else if (k==2){
          train_sub=rbind(fold1,fold2,fold3,fold5); test_sub=fold4
        } else if (k==3){
          train_sub=rbind(fold1,fold2,fold4,fold5); test_sub=fold3
        } else if (k==4){
          train_sub=rbind(fold1,fold3,fold4,fold5); test_sub=fold2
        } else if (k==5){
          train_sub=rbind(fold2,fold3,fold4,fold5); test_sub=fold1
        }
        
        #first, run a linear model on the sub training set
        lm = lm(WomacTotal_Delta ~ treat, data=train_sub, weights=IPW)
        
        tree <- pmtree(lm,data=train_sub,
                       control = ctree_control(maxdepth = depth, alpha = p))
        
        # predicted response in test set
        tryCatch({
          test_sub$node <- predict(tree, type = "node", newdata = test_sub)
          test_sub$pred <- predict(tree, type = "pass", newdata = test_sub)
          
          results[i,k+2+reps*10] = sqrt(colMeans((test_sub$pred-test_sub$WomacTotal_Delta)**2))
          results[i,k+7+reps*10] = colMeans(abs(test_sub$pred-test_sub$WomacTotal_Delta))
        }, error=function(e){"ERROR: new factor levels detected in test set"})
      }
      i=i+1
      print(paste0("Repetition=",reps,"; Depth=",depth,"; alpha=",p))
    }
  }
}
toc() #took 72 minutes

#calculate average RMSE & MAE across folds
results$rmse = rowMeans(results[,c(13:17,23:27,33:37,43:47,53:57,63:67,73:77,83:87,93:97,103:107)],na.rm=TRUE)
results$mae = rowMeans(results[,c(18:22,28:32,38:42,48:52,58:62,68:72,78:82,88:92,98:102,108:112)],na.rm=TRUE)

results2 = results %>% select(c(depth,p,rmse,mae))

saveRDS(results2, 'tree results.RData')
results2 = readRDS('tree results.RData')

bestRMSE = sqldf('select *
                 from results2
                 where rmse in 
                  (select min(rmse)
                  from results2)')

bestMAE = sqldf('select *
                 from results2
                 where mae in 
                  (select min(mae)
                  from results2)')

lm = lm(WomacTotal_Delta ~ treat, data=df_complete, weights=IPW)

tree <- pmtree(lm,data=df_complete %>% dplyr::select(-c(STUDYID)),
                 control = ctree_control(maxdepth = 4, alpha = 0.3))

plot(tree, terminal_panel = node_pmterminal(tree, plotfun = lm_plot))

#standard deviation = benchmark for RMSE
sd(df_complete$WomacTotal_Delta)

#average absolute deviation = benchmark for MAE
sqldf('select avg(diff)
      from
        (select abs(WomacTotal_Delta-mean) as diff
        from df_complete, 
          (select avg(WomacTotal_Delta) as mean
          from df_complete))')

##################################################
# re-running models with split to generate plots #
##################################################

library(grid)

df_complete$bl44 = with(df_complete, ifelse(use_WomacTotal_Pre<=44,0,1))

table(df_complete$bl44)
prop.table(table(df_complete$bl44))

summary(df_complete[df_complete$bl44==0 & df_complete$treat!='Control','WomacTotal_Delta'])
summary(df_complete[df_complete$bl44==1 & df_complete$treat!='Control','WomacTotal_Delta'])

length(df_complete[df_complete$bl44==1 & df_complete$treat!='Control','WomacTotal_Delta'])

lm2 = lm(WomacTotal_Delta ~ treat+bl44+treat*bl44,
         data=df_complete,
         weights=IPW)

table(df_complete$bl44,df_complete$treat)

posthoc = emmeans(lm2, 
                  specs=c("treat","bl44"), 
                  at=list(bl44=c(0,1))) %>% data.frame %>%
  mutate(treat2 = c('Control','Group PT','Individual PT','STEP KOA','Control','Group PT','Individual PT','STEP KOA')) %>%
  mutate('Baseline WOMAC Score' = ifelse(bl44==0,'≤44','>44')) %>%
  mutate(sample_size = c("(n=46)","(n=81)","(n=79)","(n=84)","(n=61)","(n=72)","(n=68)","(n=130)"))

contrast(emmeans(lm2, ~treat|bl44), interaction="pairwise",adjust="tukey")

ggplot(data=posthoc, aes(y=emmean, x=treat2, ymin=lower.CL, ymax=upper.CL, 
                         color=posthoc[,'Baseline WOMAC Score'], 
                         shape=posthoc[,'Baseline WOMAC Score'],
                         group=posthoc[,'Baseline WOMAC Score'])) +
  geom_point(stat="identity", size=3, position=position_dodge(width=-0.6)) +
  geom_errorbar(width=0.1, position=position_dodge(width=-0.6)) + 
#  scale_y_continuous(breaks=seq(-10,6,2)) +
  xlab("") +
#  scale_x_discrete(labels = posthoc$sample_size) +
  ylab("WOMAC Change Score") +
  guides(shape=guide_legend(title="Baseline WOMAC Score:",reverse=TRUE),
         color=guide_legend(title="Baseline WOMAC Score:",reverse=TRUE))+
  theme_classic()+
  theme(legend.position="bottom") +
  annotate("text",1-.15,1,label="n=46",size=3) + 
  annotate("text",1+.15,3.5,label="n=61",size=3) +
  annotate("text",2-.15,-9,label="n=81",size=3) +
  annotate("text",2+.15,-6,label="n=72",size=3) +
  annotate("text",3-.15,-5.5,label="n=79",size=3) +
  annotate("text",3+.15,-4,label="n=68",size=3) +
  annotate("text",4-.15,-5.5,label="n=84",size=3) +
  annotate("text",4+.15,-4,label="n=130",size=3) 

ggplot(data=posthoc, aes(y=emmean, x=posthoc[,'Baseline WOMAC Score'], ymin=lower.CL, ymax=upper.CL, color=posthoc[,'Baseline WOMAC Score'], group=treat2)) +
  geom_point(stat="identity", size=3, position=position_dodge(width=0.6)) +
  geom_errorbar(width=0.1, position=position_dodge(width=0.6)) + 
  #  scale_y_continuous(breaks=seq(-10,6,2)) +
  scale_x_discrete(breaks = posthoc$treat2, labels=posthoc$treat2) +
  xlab("") +
  ylab("WOMAC Change Score") +
  guides(color=guide_legend(title="Baseline WOMAC Score:",reverse=TRUE))+
  theme_classic()+
  theme(legend.position="bottom")

ggsave("CATE figure 1-1-25.jpeg", width=7.5, height=3.5, dpi=600)

dev.off()


#####################################################

lm2a = lm(WomacTotal_Delta ~ treat,
         data=df_complete,
         weights=IPW,
         subset=bl44==0)

posthoc2a = emmeans(lm2a, 
                  specs=c("treat")) %>% data.frame
