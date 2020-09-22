library(CytoML)
library(flowWorkspace)
library(flowCore)
library(Rphenoannoy)

# Get popID annotations from flowjo workspace and set improprely gated cells as ungated
get_fj_annotation_leafs<-function(wsp,popID=NULL,root=""){
    require(CytoML)
    require(flowWorkspace)
    ws<-open_flowjo_xml(wsp)

    gs<-flowjo_to_gatingset(ws,name=1)
    N<-length(gs)
    out<-list()[1:N]
    if (N>1) {
        samptab<-fj_ws_get_samples(ws)
        names(out)<-samptab$name
    }
    for (gsi in 1:N){
        PP<-gs_get_pop_paths(gs[[gsi]])
        PP<-PP[grep(root,PP)]
        ##choose leafs
        leafs<-NULL
        for (ii in 1:length(PP)) if (length(grep(PP[ii],PP,fixed=TRUE))==1) leafs<-c(leafs,ii)
        if (root=="") leafs<-leafs[-1]

        indM<-NULL

        for (ii in leafs) indM<-cbind(indM,as.numeric(gh_pop_get_indices_mat(gs[[gsi]],PP[ii])))

        ss<-which(rowSums(indM)>1)
        if (length(ss)>0){
            cat(length(ss), "cells found in multiple gates. Set as ungated\n")
            indM[ss,]<-0
        }
        gates<-unique(gsub(".*/","",PP[leafs]))

        g<-gsub(".*/","",PP[leafs])
        indMu<-NULL
        ##merge gates of same name
        for (i in gates){
            ss<-which(g==i)
            if (length(ss)>1) indMu<-cbind(indMu,rowSums(indM[,ss])) else indMu<-cbind(indMu,indM[,ss])
        }

        colnames(indMu)<-gates
        
        gate_ev<-colnames(indMu)[apply(indMu,MARGIN=1,which.max)]
        ss<-which(rowSums(indMu)==0)
        gate_ev[ss]<-"ungated"

        if (!is.null(popID)){
            ss<-which(!(gates %in% names(popID)))

            if (length(ss)>0){
                cat("Gates not in popID: ",gates[ss], "\n")
                ss<-which(gate_ev %in% gates[ss])
            }

            if (length(ss)>0){
                cat("Missing gates set as ungated. #Cells: ",length(ss), "\n")
                gate_ev[ss]<-"ungated"
            }

            gate_ev<-as.numeric(popID[gate_ev])
        } else gate_ev<-as.factor(gate_ev)
        out[[gsi]]<-gate_ev
    }
    if (N==1) return(gate_ev) else return(out)
}

# Equal sampling of all populations, except ungated which is oversampled
sample_eq<-function(gate,N=10000,replace=TRUE,ungated=NULL){
    res<-NULL
    if (is.null(ungated)) ungated <- -1
    for (i in unique(gate)){
        res<-c(res,base::sample(x=which(gate==i),size=N,replace=replace))
    }
    return(res[base::sample(length(res))])
}

# Prepare data for training
preprocess_file_train<-function(fcss,wsps,channels=NULL,popID=NULL, transform=FlowCIPHE::logicle.CIPHE,value=NULL,comp="SPILL",pop2cluster="ungated",K=5,N=10000,cluster_fun="kmeans", use_scatter = F){
    # Compensate, transform and concatenate files and associated gates
    gate<-data<-NULL
    for (i in 1:length(fcss)){
        fcs<-flowCore::read.FCS(fcss[i])
        fcs<-flowCore::compensate(fcs,fcs@description[[comp]])
        fcs<-transform(fcs,value=value)
        if(use_scatter) fcs <- normalize_scatter(fcs, to_normalize = channels[1:2], channels = channels[3:17])
        
        if(length(wsps)==1){
            if(i==1){
                gates <- get_fj_annotation_leafs(wsps,popID,root="Live2/|Beads") #eviter de chercher les leafs à chaque fois
            }
        }else{
            gate_i<-get_fj_annotation_leafs(wsps[i],popID)
        }
        if(is.null(channels)) channels<-1:ncol(exprs(fcs))
        data<-rbind(data,exprs(fcs)[,channels])
        ## data[,1]<-data[,1]/10000
        if(length(wsps)>1){
            gate<-c(gate,gate_i)
        }
    }
    if(length(wsps)==1){
        gate <- unlist(gates)
    }

    if (!is.null(pop2cluster)){
        if (pop2cluster %in% names(popID)) ungated<-popID[pop2cluster] else ungated<-NULL 
    }else{  
        ungated<-NULL
    }
    ##return(list(gate,out))
    out <- list()[1:3]
    names(out) <- c("data","gate","nonId")
    if (!is.null(pop2cluster))
        if (pop2cluster %in% names(popID)){
            ## k <- kmeans(out$data[which(out$gate==which(names(popID)==pop2cluster)),],K)$cluster
            
            if(cluster_fun=="Rphenoannoy"){
                Rpheno_out <- Rphenoannoy(data[which(gate==which(names(popID)==pop2cluster)),],k=50) # Build clusters with knn, k=50
                k <- membership(Rpheno_out[[2]])
            }
            else if(cluster_fun=="kmeans"){
                k <- kmeans(data[which(gate==which(names(popID)==pop2cluster)),],K)$cluster # Build K clusters from ungated cells
            }
            k[k!=1]<-k[k!=1]+max(gate)-1
            k[k==1]<-which(names(popID)==pop2cluster)
            # res <- NULL
            # for (clust in unique(k)){
            #     res<-c(res,base::sample(x=which(k==clust),size=N,replace=replace))
            # }
            
            gate[gate==which(names(popID)==pop2cluster)]<-k # Update gate with new clusters
            out[["nonId"]] <- sort(unique(k))

        }

    shufsample<-sample_eq(gate,ungated=ungated,N=N)
    out$data <- data[shufsample,]
    out$gate <- gate[shufsample]
    return(out)
}

# Prepare data for annotation
preprocess_file_annot<-function(fcs,channels=NULL,transform=FlowCIPHE::logicle.CIPHE,value=NULL,comp="SPILL",use_scatter=F){
    fcs<-flowCore::read.FCS(fcs)
    fcs<-flowCore::compensate(fcs,fcs@description[[comp]])
    fcs<-transform(fcs,value=value)
    if(use_scatter) fcs <- normalize_scatter(fcs, to_normalize = channels[1:2], channels = channels[3:17])
    if(is.null(channels)) channels<-1:ncol(exprs(fcs))
    out<-exprs(fcs)[,channels]
    ## out[,1]<-out[,1]/10000
    out
}

build_and_train<-function(fcss,wsps,channels=NULL,popID=NULL, transform=FlowCIPHE::logicle.CIPHE,value=NULL,comp="SPILL",pop2cluster="ungated",K=5,epochs=20L,patience=10L,N=10000,encoder=c(128,128,128),drate=c(0.5,0.5,0.2), use_scatter = F){
    train<-preprocess_file_train(fcss,wsps,channels,popID, transform,value,comp,pop2cluster,K,N,use_scatter = use_scatter)
    annot_machine=Annotation(train$data,train$gate,epochs=as.integer(epochs),patience=as.integer(patience),encoder=as.integer(encoder),drate=drate)
    annot_machine[["nonId"]]<-train$nonId
    return(annot_machine) # [[1]] predicted gates [[2]] network [[3]] forward pass function [["nonId"]] populations of ungated clusters 
}

# Normal training + binary classification between each pop and ungated
build_and_train_M<-function(fcss,wsps,channels=NULL,popID=NULL, transform=FlowCIPHE::logicle.CIPHE,value=NULL,comp="SPILL",pop2cluster="ungated",K=5,epochs=1L,patience=10L,N=10000,encoder=c(128,128,128),drate=c(0.5,0.5,0.2)){
    out<-list()[1:(length(popID)+1)]
    train<-preprocess_file_train(fcss,wsps,channels,popID, transform,value,comp,pop2cluster,K,N)
    annot_machine=Annotation(train$data,train$gate,epochs=as.integer(epochs),patience=as.integer(patience),encoder=as.integer(encoder),drate=drate)
    annot_machine[["nonId"]]<-train$nonId
    out[[1]]<-annot_machine
    j<-1
    for (i in sort(as.numeric(popID))){
        j<-j+1
        if (i==11) next
        print(paste("training:", i,"\n"))
        ss<-which(train$gate %in% c(i,11))
        gateLoc<-train$gate[ss]
        gateLoc[gateLoc!=i]<-0
        gateLoc[gateLoc==i]<-1
        GG<<-gateLoc
        SS<<-ss
        shufsample<-sample_eq(gateLoc,ungated=NULL,N=50000)
        out[[j]]<-Annotation(train$data[ss[shufsample],],gateLoc[shufsample],epochs=as.integer(epochs),patience=as.integer(patience),encoder=as.integer(encoder),drate=drate)
    }

    return(out)
}

# Enrich fcs file with neuralnet annotations
annotate_file<-function(fcs,machine,channels=NULL,transform=FlowCIPHE::logicle.CIPHE,value=NULL,comp="SPILL",file=NULL,use_scatter=F){

    if (!is.null(file)) ff<-fcs
    fcs<-preprocess_file_annot(fcs,channels,transform,value,comp,use_scatter=use_scatter)
    fcs<-machine[[3]](fcs)
    
    if (!is.null(machine$nonId)) fcs[fcs %in% machine$nonId]<-machine$nonId[1] # Merge ungated clusters into "ungated"
    
    if (!is.null(file)){
        ff<-flowCore::read.FCS(ff)
        fcso<- FlowCIPHE::enrich.FCS.CIPHE(ff,fcs,"auto_annot")
        flowCore::write.FCS(fcso,filename = file)
    }
    fcs
}

annotate_file_ivis<-function(fcs,machine,ivismodel,channels=NULL,transform=FlowCIPHE::logicle.CIPHE,value=NULL,comp="SPILL",file=NULL,use_scatter=F){
    
    if (!is.null(file)) ff<-fcs
    fcs<-preprocess_file_annot(fcs,channels,transform,value,comp,use_scatter=use_scatter)
    red <- ivismodel(fcs)$numpy()
    fcs<-machine[[3]](red)
    
    if (!is.null(machine$nonId)) fcs[fcs %in% machine$nonId]<-machine$nonId[1] # Merge ungated clusters into "ungated"
    
    if (!is.null(file)){
        ff<-flowCore::read.FCS(ff)
        fcso<- FlowCIPHE::enrich.FCS.CIPHE(ff,fcs,"auto_annot")
        flowCore::write.FCS(fcso,filename = file)
    }
    fcs
}

# Enrich fcs file with advanced neuralnet annotations
annotate_file_M<-function(fcs,machine,channels=NULL,transform=FlowCIPHE::logicle.CIPHE,value=NULL,comp="SPILL",file=NULL){

    if (!is.null(file)) ff<-fcs
    fcsf<-preprocess_file_annot(fcs,channels,transform,value,comp)
    fcs<-machine[[1]][[3]](fcsf)
    if (!is.null(machine[[1]]$nonId)) fcs[fcs %in% machine[[1]]$nonId]<-machine[[1]]$nonId[1] # Merge ungated clusters into "ungated"
    for (i in 1:(length(machine)-1)){

        if (i==11) next
        fcsl<-which(fcs==i)

        if (length(fcsl)>1){
            o<-machine[[i+1]][[2]](fcsf[fcsl,]) #forward pass data annotated as pop i through nn
            fcs[fcsl[o$numpy()[,2]<=0.99]]<-11 #if below threshold set as ungated
        }
    }
    if (!is.null(file)){
        ff<-flowCore::read.FCS(ff)
        fcso<- FlowCIPHE::enrich.FCS.CIPHE(ff,fcs,"annot")
        flowCore::write.FCS(fcso,filename = file)
    }

    fcs
}

# Compute precision, recall, F1
PRF1<-function(y_pred,y_true,popID=NULL){
    if (is.null(popID)) lev<-sort(union(unique(y_pred),unique(y_true))) else lev<-sort(as.integer(popID))
    tab<-table(factor(y_pred, levels = lev), factor(y_true, levels = lev))
    col_precision = diag(tab)/rowSums(tab)
    col_recall = diag(tab)/colSums(tab)
    F1<-2/(1/col_precision+1/col_recall)

    if (is.null(popID)) popID=lev else popID<-names(popID)
    return(data.frame(pop=popID,precision=col_precision,recall=col_recall,F1=F1,realsize=colSums(tab),predicted=rowSums(tab)))
}

normalize_scatter <- function(fcs, to_normalize = 1, channels=NULL){
    fcse <- fcs@exprs
    
    new_mean <-  mean(apply(fcse[,channels],2,mean))
    new_sd <- mean(apply(fcse[,channels],2,sd))
    
    for(channel in to_normalize){
        cur_mean <- mean(fcse[,channel])
        cur_sd <- sd(fcse[,channel])
        fcse[,channel] <- (fcse[,channel] - cur_mean) * (new_sd/cur_sd) + new_mean
    }
    fcs@exprs <- fcse
    return(fcs)
}