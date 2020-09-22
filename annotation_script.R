reticulate::source_python("python/annotation.py")
source("init_data.R")
source("dyadem_helpers.R")

#####
# Normal annotation
#####

annot <- function(channels, use_scatter = F){
    if(use_scatter) channels <- as.character(c(fcs@parameters@data$name[c(1,4)],channels))
    fcss <- training_fcss
    wsps <- training_wsps
    mach <- build_and_train(fcss,wsps,channels=channels,popID = popID,pop2cluster="ungated",K=5L,N=10000L,epochs=25L,value = 500L,encoder = c(128,128,128,128),drate=c(0.2,0.2,0.2,0.2), use_scatter=use_scatter)
    
    Basal <- annotate_file(validation_fcss["Basal"],mach,channels,file ="./auto_annoted/enriched_NotCST_FMO_FMO_FMO_FMO_basal_19-JUL-2018_1204_mice#41_filter-N.fcs_enrich_annotated_ciphe.fcs" ,value=500, use_scatter=use_scatter)
    aCd3 <- annotate_file(validation_fcss["aCd3"],mach,channels,file ="./auto_annoted/enriched_NotCST_FMO_FMO_FMO_FMO_CD3e_24-JAN-2019_5820_mice#190_filter-N.fcs_enrich_annotated_ciphe.fcs",value=500, use_scatter=use_scatter)
    polyC <- annotate_file(validation_fcss["polyC"],mach,channels,file = "./auto_annoted/enriched_NotCST_FMO_FMO_FMO_FMO_polyIC_22-JAN-2019_5701_mice#184_filter-N.fcs_enrich_annotated_ciphe.fcs",value=500, use_scatter=use_scatter)
    
    ground <- get_fj_annotation_leafs(validation_wsp,root="Live/|Beads",popID = popID)
    
    oB <- PRF1(Basal,ground[[2]],popID)
    oB <- oB[1:27,]
    oaCd3 <- PRF1(aCd3,ground[[3]],popID)
    oaCd3 <- oaCd3[1:27,]
    oPC <- PRF1(polyC,ground[[1]],popID)
    oPC <- oPC[1:27,]
    
    gates <- list(Basal,aCd3,polyC)
    
    return(list(oB,oaCd3,oPC,ground,mach,gates))
}

no_scatter <- annot(channels)
with_scatter <- annot(channels, use_scatter = T)

#####
# Remove debris from existing annotation machine results
#####

annot_nod <- function(channels, mach, use_scatter = F){
    if(use_scatter) channels <- as.character(c(fcs@parameters@data$name[c(1,4)],channels))
    keys <- read.csv("./Good NODE_debris.csv",sep=";")[,c(3,6)]
    popID <- keys[,1]
    names(popID) <- keys[,2]
    
    Basal=annotate_file("FMO BASAL_C1_001_007_rev.fcs",mach,channels,file ="./auto_annoted/enriched_NotCST_FMO_FMO_FMO_FMO_basal_19-JUL-2018_1204_mice#41_filter-N.fcs_enrich_annotated_ciphe_rev_noD.fcs" ,value=500, use_scatter=use_scatter)
    aCd3=annotate_file("FMO CD3e_C1_001_008_rev.fcs",mach,channels,file ="./auto_annoted/enriched_NotCST_FMO_FMO_FMO_FMO_CD3e_24-JAN-2019_5820_mice#190_filter-N.fcs_enrich_annotated_ciphe_rev_noD.fcs",value=500, use_scatter=use_scatter)
    polyC=annotate_file("FMO polyIC_C1_001_009_rev.fcs",mach,channels,file = "./auto_annoted/enriched_NotCST_FMO_FMO_FMO_FMO_polyIC_22-JAN-2019_5701_mice#184_filter-N.fcs_enrich_annotated_ciphe_rev_noD.fcs",value=500, use_scatter=use_scatter)
    
    ground_basal <- get_fj_annotation_leafs("./Basal-Independant Landmarks18.07.2018-debris.wsp",root="Live/|Beads/|Debris",popID = popID)
    ground_cd3 <- get_fj_annotation_leafs("./Cd3-Independant Landmarks18.07.2018-debris.wsp",root="Live/|Beads/|Debris",popID = popID)
    ground_pC <- get_fj_annotation_leafs("./pIC-Independant Landmarks18.07.2018-debris.wsp",root="Live/|Beads/|Debris",popID = popID)
    
    ground <- list(ground_basal, ground_cd3, ground_pC)
    
    ground <- lapply(ground, function(x){
        deb <- which(x==28)
        x[deb] <- rep(-1,length(deb))
        return(x)
    })
    
    oB <- PRF1(Basal,ground[[1]],popID)
    oB <- oB[1:27,]
    oaCd3 <- PRF1(aCd3,ground[[2]],popID)
    oaCd3 <- oaCd3[1:27,]
    oPC <- PRF1(polyC,ground[[3]],popID)
    oPC <- oPC[1:27,]
    
    gates<-list(Basal,aCd3,polyC)

    return(list(oB,oaCd3,oPC,ground,gates))
}

#####
# Reverse annotation (training <=> validation), sort by flag
#####

annot_reverse <- function(channels, use_scatter = F){
    if(use_scatter) channels <- as.character(c(fcs@parameters@data$name[c(1,4)],channels))
    fcss <- validation_fcss
    wsps <- validation_wsp
    
    mach <- build_and_train(fcss,wsps,channels=channels,popID = popID,pop2cluster="ungated",K=5,N=10000,epochs=50,value = 500,encoder = c(128,128,128,128),drate=c(0.2,0.5,0.5,0.2), use_scatter=use_scatter)
    
    Basal <- annotate_file(training_fcss["Basal"],mach,channels,file ="./auto_annoted/enriched_NotCST_FMO_FMO_FMO_FMO_basal_19-JUL-2018_1204_mice#41_filter-N.fcs_enrich_annotated_ciphe_rev.fcs" ,value=500, use_scatter=use_scatter)
    aCd3 <- annotate_file(training_fcss["aCd3"],mach,channels,file ="./auto_annoted/enriched_NotCST_FMO_FMO_FMO_FMO_CD3e_24-JAN-2019_5820_mice#190_filter-N.fcs_enrich_annotated_ciphe_rev.fcs",value=500, use_scatter=use_scatter)
    polyC <- annotate_file(training_fcss["polyC"],mach,channels,file = "./auto_annoted/enriched_NotCST_FMO_FMO_FMO_FMO_polyIC_22-JAN-2019_5701_mice#184_filter-N.fcs_enrich_annotated_ciphe_rev.fcs",value=500, use_scatter=use_scatter)
    
    ground_basal <- get_fj_annotation_leafs("./Basal-Independant Landmarks18.07.2018.wsp",root="Live/|Beads",popID = popID)
    ground_cd3 <- get_fj_annotation_leafs("./Cd3-Independant Landmarks18.07.2018.wsp",root="Live/|Beads",popID = popID)
    ground_pC <- get_fj_annotation_leafs("./pIC-Independant Landmarks18.07.2018.wsp",root="Live/|Beads",popID = popID)

    ground <- list(ground_basal, ground_cd3, ground_pC)
    
    oB <- oaCd3 <- oPC <- list()[1:3]
    
    for(i in 1:3){
        bex <- read.FCS("FMO BASAL_C1_001_007_rev.fcs")@exprs
        bf <- which(bex[,"Flag"]==i)
        oB[[i]] <- PRF1(Basal[bf], ground[[1]][bf],popID)
        oB[[i]] <- oB[[i]][1:27,]
        
        cd3ex <- read.FCS("FMO CD3e_C1_001_008_rev.fcs")@exprs
        cd3f <- which(cd3ex[,"Flag"]==i)
        oaCd3[[i]] <- PRF1(aCd3[cd3f], ground[[2]][cd3f],popID)
        oaCd3[[i]] <- oaCd3[[i]][1:27,]
        
        pcex <- read.FCS("FMO polyIC_C1_001_009_rev.fcs")@exprs
        pcf <- which(pcex[,"Flag"]==i)
        oPC[[i]] <- PRF1(polyC[pcf], ground[[3]][pcf],popID)
        oPC[[i]] <- oPC[[i]][1:27,]
    }
    names(oB)<-names(oaCd3)<-names(oPC) <- c("Flag 1", "Flag 2", "Flag 3")
    
    gates<-list(Basal,aCd3,polyC)
    
    return(list(oB,oaCd3,oPC,ground,mach,gates))
}

reverse_ws <- annot_reverse(channels, use_scatter = T)
mach <- reverse_ws[[5]]
reverse_ws_noD <- annot_nod(channels, mach, use_scatter = T)

#####
# Annotation with the help of ivis dimred
#####

ivismodel <- outputs[[3]]$layers[[1]] # TODO: fix load_model function
annot_ivis <- function(channels, ivismodel, use_scatter = F){
    if(use_scatter) channels <- as.character(c(fcs@parameters@data$name[c(1,4)],channels))
    fcss <- training_fcss
    wsps <- training_wsps
    train<-preprocess_file_train(fcss,wsps,channels,popID,value=500,use_scatter = use_scatter)
    
    red <- ivismodel(train$data)$numpy()
    mach=Annotation(red,train$gate,epochs=25L,patience=10L)
    mach[["nonId"]]<-train$nonId
    
    Basal <- annotate_file_ivis(validation_fcss["Basal"],mach,ivismodel,channels,file ="./auto_annoted/enriched_NotCST_FMO_FMO_FMO_FMO_basal_19-JUL-2018_1204_mice#41_filter-N.fcs_enrich_annotated_ciphe.fcs" ,value=500, use_scatter=use_scatter)
    aCd3 <- annotate_file_ivis(validation_fcss["aCd3"],mach,ivismodel,channels,file ="./auto_annoted/enriched_NotCST_FMO_FMO_FMO_FMO_CD3e_24-JAN-2019_5820_mice#190_filter-N.fcs_enrich_annotated_ciphe.fcs",value=500, use_scatter=use_scatter)
    polyC <- annotate_file_ivis(validation_fcss["polyC"],mach,ivismodel,channels,file = "./auto_annoted/enriched_NotCST_FMO_FMO_FMO_FMO_polyIC_22-JAN-2019_5701_mice#184_filter-N.fcs_enrich_annotated_ciphe.fcs",value=500, use_scatter=use_scatter)
    
    ground <- get_fj_annotation_leafs(validation_wsp,root="Live/|Beads",popID = popID)
    
    oB <- PRF1(Basal,ground[[2]],popID)
    oB <- oB[1:27,]
    oaCd3 <- PRF1(aCd3,ground[[3]],popID)
    oaCd3 <- oaCd3[1:27,]
    oPC <- PRF1(polyC,ground[[1]],popID)
    oPC <- oPC[1:27,]
    
    gates <- list(Basal,aCd3,polyC)
    return(list(oB,oaCd3,oPC,ground,mach,gates))
}

ivis_ws <- annot_ivis(channels=channels, ivismodel=ivismodel, use_scatter=T)
