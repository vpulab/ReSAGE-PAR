
experimentId = "b5ad9cb5e11f47b0b57957305ac96bd7"
strSeed="605"
ipExperiment="192.168.23.203"
portExperiment="5000"

out

listAttributesPA100k= ['Hat','Glasses','ShortSleeve','LongSleeve','UpperStride','UpperLogo','UpperPlaid','UpperSplice','LowerStripe','LowerPattern','LongCoat','Trousers','Shorts','Skirt&Dress','boots', 'HandBag','ShoulderBag','Backpack','HoldObjectsInFront','AgeOver60','Age18-60','AgeLess18','Female','Front','Side','Back']
listOfAttributes = listAttributesPA100k




# plot training vs testing, url for plot in attributeAnalysis.txt
metricsTrainingVSTesting=[]
# plot general testing, url for plot in attributeAnalysis.txt
metricsGeneralTesting = []
# plot general training, url for plot in attributeAnalysis.txt
metricsGeneralTraining = []
# plot training labels, url for plot in attributeAnalysis.txt
metricsTrainingLabels = []
# plot testing labels, url for plot in attributeAnalysis.txt
metricsTestingLabels = []
# plot training instance, url for plot in attributeAnalysis.txt
metricsTrainingInstance = []
# plot testing instance, url for plot in attributeAnalysis.txt
metricsTestingInstance = []

for attribute in listOfAttributes:
    metricsTrainingVSTesting.append("training_ma_"+strSeed)
    metricsTrainingVSTesting.append("training_ma_"+strSeed)

    metricsGeneralTesting.append("testing_ma_"+strSeed)
    metricsGeneralTesting.append("testing_label_f1_"+strSeed)
    metricsGeneralTesting.append("testing_pos_recall_"+strSeed)
    metricsGeneralTesting.append("testing_neg_recall_"+strSeed)
    metricsGeneralTesting.append("testing_acc_"+strSeed)
    metricsGeneralTesting.append("testing_prec_"+strSeed)
    metricsGeneralTesting.append("testing_rec_"+strSeed)
    metricsGeneralTesting.append("testing_f1_"+strSeed)

    metricsGeneralTraining.append("training_ma_"+strSeed)
    metricsGeneralTraining.append("training_label_f1_"+strSeed)
    metricsGeneralTraining.append("training_pos_recall_"+strSeed)
    metricsGeneralTraining.append("training_neg_recall_"+strSeed)
    metricsGeneralTraining.append("training_acc_"+strSeed)
    metricsGeneralTraining.append("training_prec_"+strSeed)
    metricsGeneralTraining.append("training_rec_"+strSeed)
    metricsGeneralTraining.append("training_f1_"+strSeed)

    metricsTrainingLabels.append("training_"+attribute+"_ma_"+strSeed)
    metricsTrainingLabels.append("training_"+attribute+"_label_f1_"+strSeed)
    metricsTrainingLabels.append("training_"+attribute+"_label_pos_recall_"+strSeed)
    metricsTrainingLabels.append("training_"+attribute+"_label_neg_recall_"+strSeed)
    metricsTrainingLabels.append("training_"+attribute+"_label_acc_"+strSeed)
    metricsTrainingLabels.append("training_"+attribute+"_label_prec_"+strSeed)

    metricsTrainingInstance.append("training_"+attribute+"_instance_acc_label_"+strSeed)
    metricsTrainingInstance.append("training_"+attribute+"_instance_prec_label_"+strSeed)
    metricsTrainingInstance.append("training_"+attribute+"_instance_recall_label_"+strSeed)
    metricsTrainingInstance.append("training_"+attribute+"_instance_f1_label_"+strSeed)

    metricsTestingLabels.append("testing_"+attribute+"_ma_"+strSeed)
    metricsTestingLabels.append("testing_"+attribute+"_label_f1_"+strSeed)
    metricsTestingLabels.append("testing_"+attribute+"_label_pos_recall_"+strSeed)
    metricsTestingLabels.append("testing_"+attribute+"_label_neg_recall_"+strSeed)
    metricsTestingLabels.append("testing_"+attribute+"_label_acc_"+strSeed)
    metricsTestingLabels.append("testing_"+attribute+"_label_prec_"+strSeed)
                            
    metricsTestingInstance.append("testing_"+attribute+"_instance_acc_label_"+strSeed)
    metricsTestingInstance.append("testing_"+attribute+"_instance_prec_label_"+strSeed)
    metricsTestingInstance.append("testing_"+attribute+"_instance_recall_label_"+strSeed)
    metricsTestingInstance.append("testing_"+attribute+"_instance_f1_label_"+strSeed)






metricsTrainingLabels




metricsToPlot = []



str1="http://192.168.23.203:5000/#/metric/learning_rate?runs=[\"b5ad9cb5e11f47b0b57957305ac96bd7\"]&experiments=[\"9\"]&plot_metric_keys=[\""

#testing_ma_605","training_ma_605","learning_rate","testing_Age18-60_label_acc_605

str2="\"]&plot_layout={\"autosize\":true,\"xaxis\":{\"autorange\":true,\"type\":\"linear\"},\"yaxis\":{}}&x_axis=step&y_axis_scale=linear&line_smoothness=1&show_point=false&deselected_curves=[]&last_linear_y_axis_range=[]"