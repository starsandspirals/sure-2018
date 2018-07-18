library(data.table)
library(ggplot2)

## read in data
D <- fread('person-output-0.csv')
names(D)
D[,acat:=cut(age,breaks=c(0,5,15,25,35,45,55,65,Inf),include.lowest = TRUE)]
D[,acat2:=cut(age,breaks=c(0,17,25,45,Inf),include.lowest = TRUE)]
D[,sexage:=paste0(gender,':',acat)]
tmp <- D[,.N,by=.(household_size,sexage)]
tmp

ggplot(D,aes(x=acat)) + geom_bar()


## comparing household size distribution
DM <- dcast(tmp[,.(N,sexage,hhsize=household_size)],sexage~hhsize,value.var='N')
DM
dm <- as.matrix(DM[,-1,with=FALSE])
print(sum(dm,na.rm=TRUE))
(dm <- dm/sum(dm,na.rm=TRUE))

H <- fread('histo.csv',header=TRUE)
h <- as.matrix(H[,-1,with=FALSE])
print(sum(h))
(h <- h/sum(h))
HM <- melt(H)
names(HM)[2:3] <- c('hhsize','hcount')
str(HM)
HM$hhsize <- as.integer(as.character(HM$hhsize))


## via shape (not robust)
qplot(x=c(dm),y=c(h),geom='point') + coord_fixed() + geom_abline(intercept=0,slope=1,col=2)
ggsave('hh1.pdf')


HM
names(tmp) <- c('hhsize','sexage','hcountS')
str(tmp)

## via merge in case cols different
tmp <- merge(tmp,HM,by=c('hhsize','sexage'),all=TRUE)
tmp[,hcountn:=hcount/sum(hcount,na.rm=TRUE)]
tmp[,hcountSn:=hcountS/sum(hcountS,na.rm=TRUE)]
ggplot(tmp,aes(hcountn,hcountSn,col=hhsize)) + geom_point() + coord_fixed() + geom_abline(intercept=0,slope=1,col=2)
## ggplot(tmp,aes(hcountn,hcountSn,col=sexage)) + geom_point() + coord_fixed() + geom_abline(intercept=0,slope=1,col=2)
ggsave('hh2.pdf')

ggplot(tmp,aes(hcountn,1-hcountn/hcountSn,col=sexage)) + geom_point()+geom_abline(intercept=0,slope=0,col=2)



## HIV/ART (perfect)
D[,sum(hiv)]/nrow(D)
D[,sum(art)/sum(hiv)]
ggplot(D,aes(age,hiv)) + geom_rug() + geom_smooth()


## church/home etc
## total time? (perfect)
D[,alltime:=time_home + time_church + time_transport + time_visiting + time_clinic + time_workplace + time_bar + time_school + time_outside  ]
D[,summary(alltime)]


## church no?
D[,chyn:=as.integer(time_church>0)]
ggplot(D,aes(household_size,chyn)) + geom_smooth()

## time barcharts
DT <- melt(D[,.(ID,gender,acat,time_home,time_church,time_transport)],id=c('ID','gender','acat'))

ggplot(DT,aes(x=acat,y=value,fill=variable)) + geom_bar(stat='identity') + facet_wrap(~gender)

DTS <- DT[,.(value=sum(value)),by=.(gender,acat,variable)]
DTS[,prop:=value/sum(value),by=.(gender,acat)]
DTS
DTS[,sum(prop),by=.(gender,acat)]

ggplot(DTS,aes(x=acat,y=prop,fill=variable)) + geom_bar(stat='identity') + facet_wrap(~gender) + scale_y_sqrt() #ODD?


## TB
D[,sum(active_tb)]/nrow(D[age>=18])
D[,sum(active_tb)]/nrow(D)

1e5*D[,sum(active_tb)]/nrow(D)
ceiling(0.5e-2*nrow(D))

ggplot(D[,.(tb=mean(active_tb)),by=.(gender,acat,hiv,art)],aes(acat,tb)) + geom_point() + facet_grid(gender ~ hiv+art)
ggsave('tbprev.pdf')

tmp <- D[,.(tb=mean(active_tb)),by=.(hiv,art)]
tmp
tmp[,tb/tb[1]]

D[,sum(art)]

(tmp <- D[,.(tb=mean(active_tb)),by=.(gender,acat2)])
tmp[acat2!='[0,17]',.(RR=tb/tb[1],acat2,gender)]

(tmp2 <- D[,.(tb=mean(active_tb)),by=.(gender,acat2,hiv)])
tmp2[acat2!='[0,17]',.(RR=tb/tb[1],acat2,gender,hiv)]
