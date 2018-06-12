setwd('/Users/pjd/Documents/teaching/projects/2018SURE')
D <- fread('Zpop.dat')
str(D)
D[,unique(gender)]
D[gender=="" | gender=="BLANK",gender:="1"]

D[,unique(age)]

D[,acat:=cut(age,breaks=c(0,5,15,25,35,45,55,65,Inf),include.lowest = TRUE)]
D[,unique(acat)]

