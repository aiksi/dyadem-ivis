keys<-read.csv("./data/Good NODE_verif.csv",sep=";")[,c(3,6)]
popID<-keys[,1]
names(popID)<-keys[,2]
fcs<-read.FCS("./data/enriched/enriched_NotCST_FMO_FMO_FMO_FMO_CD3e_24-JAN-2019_5820_mice#190_filter-N.fcs_enrich.fcs")
channels<-as.character(fcs@parameters@data$name[!is.na(fcs@parameters@data$desc)])[1:15]

conditions = c("Basal", "aCd3", "polyC")

training_fcss = c("./data/FMO BASAL_C1_001_007_rev.fcs", 
                  "./data/FMO CD3e_C1_001_008_rev.fcs", 
                  "./data/FMO polyIC_C1_001_009_rev.fcs")
names(training_fcss) = conditions
training_wsps = c("./data/Basal-Independant Landmarks18.07.2018.wsp", 
                  "./data/Cd3-Independant Landmarks18.07.2018.wsp", 
                  "./data/pIC-Independant Landmarks18.07.2018.wsp")
names(training_wsps) = conditions

validation_fcss = c("./data/enriched/enriched_NotCST_FMO_FMO_FMO_FMO_basal_19-JUL-2018_1204_mice#41_filter-N.fcs_enrich.fcs",
                    "./data/enriched/enriched_NotCST_FMO_FMO_FMO_FMO_CD3e_24-JAN-2019_5820_mice#190_filter-N.fcs_enrich.fcs",
                    "./data/enriched/enriched_NotCST_FMO_FMO_FMO_FMO_polyIC_22-JAN-2019_5701_mice#184_filter-N.fcs_enrich.fcs")
names(validation_fcss) = conditions
validation_wsp = "./data/Validation-Perceptron.V2.wsp"