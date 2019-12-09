setwd("D:\\Tutorials\\R\\R_Codes")

rm(list = ls(all.names = TRUE)) # Clear loaded data
dev.off() # Reset graphical parameters to default

# load the library
library(rpart)
# load the dataset
data("Sonar")

D_Tree = rpart(Class ~ ., 
                  data = Sonar, 
                  cp = 10^(-6))

###### the function arguments:
# 1) formula, of the form: outcome ~ predictors
# note: outcome ~ . is 'use all other variables in data'
# 2) data: a data.frame object, or any matrix which has variables as
# columns and observations as rows
# 3) cp: used to choose depth of the tree, we'll manually prune the tree
# later and hence set the threshold very low (more on this later)
# The commands, print() and summary() will be useful to look at the tree.
# But first, lets see how big the created tree was
# The object spac.tree is a list with a number of entires that can be
# accessed via the $ symbol. A list is like a hash table.
# To see the entries in a list, use names()

D_Tree
plot(D_Tree)
summary(D_Tree)
names(D_Tree)
D_Tree$cptable # Check the size f the tree, i.e number of splits (nodes), here we get 5
# Prune the tree to 3 splits
cp3 = which(D_Tree$cptable[, 2] == 3)
D_Tree3 = prune(D_Tree, D_Tree$cptable[cp3, 1])
# now lets look at the tree with print() and summary()
D_Tree3
summary(D_Tree3)
plot(D_Tree3)

# lets get a graphical representation of the tree, and save to a
# png file
png("D_Tree.png", 
    width = 1200, 
    height = 800)
post(D_Tree, 
     file = "", 
     title. = "Classifying Sonar Mines and Rocks, 6 splits",
     bp = 18)
dev.off()


png("D_Tree3.png", 
    width = 1200, 
    height = 800)
post(D_Tree3, 
     file = "", 
     title. = "Classifying Sonar Mines and Rocks, 3 splits",
     bp = 18)
dev.off()








