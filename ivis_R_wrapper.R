library(flowCore)
library(reticulate)
library(ggplot2)
library(RColorBrewer)
library(prodlim)
library(rgl)

reticulate::source_python("python/knn_annoy.py")
reticulate::source_python("python/ivis_supervised.py")
source("init_data.R")
source("dyadem_helpers.R")

scatter <- as.character(c(fcs@parameters@data$name[c(1,4)],channels)) # comment if scatter is unwanted

train <- preprocess_file_train(training_fcss, training_wsps, channels=scatter,popID=popID,transform=FlowCIPHE::logicle.CIPHE,value=500L,comp="SPILL",K=5,N=10000,cluster_fun="kmeans", use_scatter = T)

dataset <- train$data
classes <- train$gate

shuff<-base::sample(nrow(dataset))
data<-dataset[shuff,]
cl<-classes[shuff]

print(system.time(outputs <- R_wrapper_interface(data,
                                                 cl,
                                                 epochs = 15L,
                                                 epochs_without_improvement = 4L,
                                                 start_imp = 50L,
                                                 batch_size = 8192L,
                                                 dense_layers = c(128,128,128),
                                                 alpha = c(0.1,0.1),
                                                 embedding_dim = 6L,
                                                 k = 64L,
                                                 approx = T,
                                                 distance = 'euclidean',
                                                 margin = 1.,
                                                 pn_weight = 0.1,
                                                 verbose = 1,
                                                 debug = T)))

red <- outputs[[2]]

embed <- data.frame(red)
embed <- cbind(embed, cl)
colnames(embed) <- c("ivis1","ivis2","col")

#full dimred plot
colorCount = 27
getPalette = colorRampPalette(brewer.pal(9, "Set1"))

ggplot() + 
  geom_point(data=embed, aes(x = ivis1, y = ivis2, color = as.factor(col)), shape=46) + 
  scale_color_manual(values = c(getPalette(colorCount),"#000000"))