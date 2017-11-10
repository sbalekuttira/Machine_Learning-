import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean

#total number of violations of first month for each type of violation
datakey = pd.read_csv('datatype0.csv', sep="\t", header = None,encoding='utf-8')
datakey.columns = ["eid","pii","pci","intd","dd","sc","invd","prd"]
rowcountkey=len(datakey.index)

pii0_sum=0     #pii is PersonalIdentity info
pci0_sum=0	#pci is Personal Credit info
intd0_sum=0	#intd is Internal Docs
dd0_sum=0	#dd is Design Docs
sc0_sum=0	#sc is Source Code
invd0_sum=0	#invd is Inventory Disclosure
prd0_sum=0	#prd is Program restricted Data

for i in range(rowcountkey):
	pii0_sum=datakey.pii[i]+pii0_sum
	pci0_sum=datakey.pci[i]+pci0_sum
	intd0_sum=datakey.intd[i]+intd0_sum
	dd0_sum=datakey.dd[i]+dd0_sum
	sc0_sum=datakey.sc[i]+sc0_sum
	invd0_sum=datakey.invd[i]+invd0_sum
	prd0_sum=datakey.prd[i]+prd0_sum
	

datakey = pd.read_csv('datatype1.csv', sep="\t", header = None,encoding='utf-8')
datakey.columns = ["eid","pii","pci","intd","dd","sc","invd","prd"]
rowcountkey=len(datakey.index)

pii1_sum=0
pci1_sum=0
intd1_sum=0
dd1_sum=0
sc1_sum=0
invd1_sum=0
prd1_sum=0

for i in range(rowcountkey):
	pii1_sum=datakey.pii[i]+pii1_sum
	pci1_sum=datakey.pci[i]+pci1_sum
	intd1_sum=datakey.intd[i]+intd1_sum
	dd1_sum=datakey.dd[i]+dd1_sum
	sc1_sum=datakey.sc[i]+sc1_sum
	invd1_sum=datakey.invd[i]+invd1_sum
	prd1_sum=datakey.prd[i]+prd1_sum

datakey = pd.read_csv('datatype2.csv', sep="\t", header = None,encoding='utf-8')
datakey.columns = ["eid","pii","pci","intd","dd","sc","invd","prd"]
rowcountkey=len(datakey.index)

pii2_sum=0
pci2_sum=0
intd2_sum=0
dd2_sum=0
sc2_sum=0
invd2_sum=0
prd2_sum=0

for i in range(rowcountkey):
	pii2_sum=datakey.pii[i]+pii2_sum
	pci2_sum=datakey.pci[i]+pci2_sum
	intd2_sum=datakey.intd[i]+intd2_sum
	dd2_sum=datakey.dd[i]+dd2_sum
	sc2_sum=datakey.sc[i]+sc2_sum
	invd2_sum=datakey.invd[i]+invd2_sum
	prd2_sum=datakey.prd[i]+prd2_sum
	

datakey = pd.read_csv('datatype3.csv', sep="\t", header = None,encoding='utf-8')
datakey.columns = ["eid","pii","pci","intd","dd","sc","invd","prd"]
rowcountkey=len(datakey.index)

pii3_sum=0
pci3_sum=0
intd3_sum=0
dd3_sum=0
sc3_sum=0
invd3_sum=0
prd3_sum=0

for i in range(rowcountkey):
	pii3_sum=datakey.pii[i]+pii3_sum
	pci3_sum=datakey.pci[i]+pci3_sum
	intd3_sum=datakey.intd[i]+intd3_sum
	dd3_sum=datakey.dd[i]+dd3_sum
	sc3_sum=datakey.sc[i]+sc3_sum
	invd3_sum=datakey.invd[i]+invd3_sum
	prd3_sum=datakey.prd[i]+prd3_sum
	

datakey = pd.read_csv('datatype4.csv', sep="\t", header = None,encoding='utf-8')
datakey.columns = ["eid","pii","pci","intd","dd","sc","invd","prd"]
rowcountkey=len(datakey.index)

pii4_sum=0
pci4_sum=0
intd4_sum=0
dd4_sum=0
sc4_sum=0
invd4_sum=0
prd4_sum=0

for i in range(rowcountkey):
	pii4_sum=datakey.pii[i]+pii4_sum
	pci4_sum=datakey.pci[i]+pci4_sum
	intd4_sum=datakey.intd[i]+intd4_sum
	dd4_sum=datakey.dd[i]+dd4_sum
	sc4_sum=datakey.sc[i]+sc4_sum
	invd4_sum=datakey.invd[i]+invd4_sum
	prd4_sum=datakey.prd[i]+prd4_sum


datakey = pd.read_csv('datatype5.csv', sep="\t", header = None,encoding='utf-8')
datakey.columns = ["eid","pii","pci","intd","dd","sc","invd","prd"]
rowcountkey=len(datakey.index)

pii5_sum=0
pci5_sum=0
intd5_sum=0
dd5_sum=0
sc5_sum=0
invd5_sum=0
prd5_sum=0

for i in range(rowcountkey):
	pii5_sum=datakey.pii[i]+pii5_sum
	pci5_sum=datakey.pci[i]+pci5_sum
	intd5_sum=datakey.intd[i]+intd5_sum
	dd5_sum=datakey.dd[i]+dd5_sum
	sc5_sum=datakey.sc[i]+sc5_sum
	invd5_sum=datakey.invd[i]+invd5_sum
	prd5_sum=datakey.prd[i]+prd5_sum

datakey = pd.read_csv('datatype6.csv', sep="\t", header = None,encoding='utf-8')
datakey.columns = ["eid","pii","pci","intd","dd","sc","invd","prd"]
rowcountkey=len(datakey.index)

pii6_sum=0
pci6_sum=0
intd6_sum=0
dd6_sum=0
sc6_sum=0
invd6_sum=0
prd6_sum=0

for i in range(rowcountkey):
	pii6_sum=datakey.pii[i]+pii6_sum
	pci6_sum=datakey.pci[i]+pci6_sum
	intd6_sum=datakey.intd[i]+intd6_sum
	dd6_sum=datakey.dd[i]+dd6_sum
	sc6_sum=datakey.sc[i]+sc5_sum
	invd6_sum=datakey.invd[i]+invd6_sum
	prd6_sum=datakey.prd[i]+prd6_sum

datakey = pd.read_csv('datatype7.csv', sep="\t", header = None,encoding='utf-8')
datakey.columns = ["eid","pii","pci","intd","dd","sc","invd","prd"]
rowcountkey=len(datakey.index)

pii7_sum=0
pci7_sum=0
intd7_sum=0
dd7_sum=0
sc7_sum=0
invd7_sum=0
prd7_sum=0

for i in range(rowcountkey):
	pii7_sum=datakey.pii[i]+pii7_sum
	pci7_sum=datakey.pci[i]+pci7_sum
	intd7_sum=datakey.intd[i]+intd7_sum
	dd7_sum=datakey.dd[i]+dd7_sum
	sc7_sum=datakey.sc[i]+sc7_sum
	invd7_sum=datakey.invd[i]+invd7_sum
	prd7_sum=datakey.prd[i]+prd7_sum

datakey = pd.read_csv('datatype8.csv', sep="\t", header = None,encoding='utf-8')
datakey.columns = ["eid","pii","pci","intd","dd","sc","invd","prd"]
rowcountkey=len(datakey.index)

pii8_sum=0
pci8_sum=0
intd8_sum=0
dd8_sum=0
sc8_sum=0
invd8_sum=0
prd8_sum=0

for i in range(rowcountkey):
	pii8_sum=datakey.pii[i]+pii8_sum
	pci8_sum=datakey.pci[i]+pci8_sum
	intd8_sum=datakey.intd[i]+intd8_sum
	dd8_sum=datakey.dd[i]+dd8_sum
	sc8_sum=datakey.sc[i]+sc8_sum
	invd8_sum=datakey.invd[i]+invd8_sum
	prd8_sum=datakey.prd[i]+prd8_sum


datakey = pd.read_csv('datatype9.csv', sep="\t", header = None,encoding='utf-8')
datakey.columns = ["eid","pii","pci","intd","dd","sc","invd","prd"]
rowcountkey=len(datakey.index)

pii9_sum=0
pci9_sum=0
intd9_sum=0
dd9_sum=0
sc9_sum=0
invd9_sum=0
prd9_sum=0

for i in range(rowcountkey):
	pii9_sum=datakey.pii[i]+pii9_sum
	pci9_sum=datakey.pci[i]+pci9_sum
	intd9_sum=datakey.intd[i]+intd9_sum
	dd9_sum=datakey.dd[i]+dd9_sum
	sc9_sum=datakey.sc[i]+sc9_sum
	invd9_sum=datakey.invd[i]+invd9_sum
	prd9_sum=datakey.prd[i]+prd9_sum


datakey = pd.read_csv('datatype10.csv', sep="\t", header = None,encoding='utf-8')
datakey.columns = ["eid","pii","pci","intd","dd","sc","invd","prd"]
rowcountkey=len(datakey.index)

pii10_sum=0
pci10_sum=0
intd10_sum=0
dd10_sum=0
sc10_sum=0
invd10_sum=0
prd10_sum=0

for i in range(rowcountkey):
	pii10_sum=datakey.pii[i]+pii10_sum
	pci10_sum=datakey.pci[i]+pci10_sum
	intd10_sum=datakey.intd[i]+intd10_sum
	dd10_sum=datakey.dd[i]+dd10_sum
	sc10_sum=datakey.sc[i]+sc10_sum
	invd10_sum=datakey.invd[i]+invd10_sum
	prd10_sum=datakey.prd[i]+prd10_sum

datakey = pd.read_csv('datatype11.csv', sep="\t", header = None,encoding='utf-8')
datakey.columns = ["eid","pii","pci","intd","dd","sc","invd","prd"]
rowcountkey=len(datakey.index)

pii11_sum=0
pci11_sum=0
intd11_sum=0
dd11_sum=0
sc11_sum=0
invd11_sum=0
prd11_sum=0

for i in range(rowcountkey):
	pii11_sum=datakey.pii[i]+pii11_sum
	pci11_sum=datakey.pci[i]+pci11_sum
	intd11_sum=datakey.intd[i]+intd11_sum
	dd11_sum=datakey.dd[i]+dd11_sum
	sc11_sum=datakey.sc[i]+sc11_sum
	invd11_sum=datakey.invd[i]+invd11_sum
	prd11_sum=datakey.prd[i]+prd11_sum

#xs is the months
xs=np.array([1,2,3,4,5,6,7,8,9,10,11,12])

#all the y's
ys_pii=np.array([pii0_sum,pii1_sum,pii2_sum,pii3_sum,pii4_sum,pii5_sum,pii6_sum,pii7_sum,pii8_sum,pii9_sum,pii10_sum,pii11_sum])

ys_pci=np.array([pci0_sum,pci1_sum,pci2_sum,pci3_sum,pci4_sum,pci5_sum,pci6_sum,pci7_sum,pci8_sum,pci9_sum,pci10_sum,pci11_sum])

ys_intd=np.array([intd0_sum,intd1_sum,intd2_sum,intd3_sum,intd4_sum,intd5_sum,intd6_sum,intd7_sum,intd8_sum,intd9_sum,intd10_sum,intd11_sum])

ys_dd=np.array([dd0_sum,dd1_sum,dd2_sum,dd3_sum,dd4_sum,dd5_sum,dd6_sum,dd7_sum,dd8_sum,dd9_sum,dd10_sum,dd11_sum])

ys_sc=np.array([sc0_sum,sc1_sum,sc2_sum,sc3_sum,sc4_sum,sc5_sum,sc6_sum,sc7_sum,sc8_sum,sc9_sum,sc10_sum,sc11_sum])

ys_invd=np.array([invd0_sum,invd1_sum,invd2_sum,invd3_sum,invd4_sum,invd5_sum,invd6_sum,invd7_sum,invd8_sum,invd9_sum,invd10_sum,invd11_sum])

ys_prd=np.array([prd0_sum,prd1_sum,prd2_sum,prd3_sum,prd4_sum,prd5_sum,prd6_sum,prd7_sum,prd8_sum,prd9_sum,prd10_sum,prd11_sum])

#hard coded Linear Regression algorithm (BEST FIT LINE), find slope and  y intercept in y=mx+b

def best_fit_and_intercept(x,y):
		m= (((mean(x)*mean(y))-mean(x*y))/
	    	((mean(x)**2)-mean(x*x)))
		b=mean(y)-m*(mean(x))
		return m,b


#x_train,x_test,y_train,y_test=cross_valintdation.train_test_split(xs,ys_email,test_size=0.2)
#clf=LinearRegression()
#clf.fit(x_train,y_train)

#dd(clf.score(x_test,y_test))

#regression line for PersonalIdentity info
m,b=best_fit_and_intercept(xs,ys_pii)
regression_line=[(m*x)+b for x in xs]

plt.scatter(xs,ys_pii,color='red')
plt.plot(xs,regression_line,color='black')
plt.title('PERSONAL IDENTITY INFORMATION VIOLATIONS')
plt.xlabel('MONTH', fontsize=18)
plt.ylabel('VIOLATIONS', fontsize=16)

#predict any month for PersonalIdentity info
predict_x=13
predict_pii_y=(m*predict_x)+b
plt.scatter(predict_x,predict_pii_y,color='blue')
plt.show()



#regression line for Personal Credit info
m,b=best_fit_and_intercept(xs,ys_pci)
regression_line=[(m*x)+b for x in xs]

plt.scatter(xs,ys_pci,color='yellow')
plt.plot(xs,regression_line,color='black')

plt.title('PERSONAL CREDIT INFORMATION VIOLATIONS')
plt.xlabel('MONTH', fontsize=18)
plt.ylabel('VIOLATIONS', fontsize=16)

#predict any month for Personal Credit info
predict_x=13
predict_pci_y=(m*predict_x)+b
plt.scatter(predict_x,predict_pci_y,color='red')
plt.show()



#regression line for Internal Docs
m,b=best_fit_and_intercept(xs,ys_intd)
regression_line=[(m*x)+b for x in xs]

plt.scatter(xs,ys_intd,color='orange')
plt.plot(xs,regression_line,color='black')

plt.title('INTERNAL DOCUMENTS VIOLATIONS')
plt.xlabel('MONTH', fontsize=18)
plt.ylabel('VIOLATIONS', fontsize=16)

#predict any month for Internal Docs
predict_x=13
predict_intd_y=(m*predict_x)+b
plt.scatter(predict_x,predict_intd_y,color='blue')
plt.show()


	
#regression line for Design Docs
m,b=best_fit_and_intercept(xs,ys_dd)
regression_line=[(m*x)+b for x in xs]

plt.scatter(xs,ys_dd,color='pink')
plt.plot(xs,regression_line,color='black')

plt.title('DESIGN DOCUMENTS VIOLATIONS')
plt.xlabel('MONTH', fontsize=18)
plt.ylabel('VIOLATIONS', fontsize=16)

#predict any month for Design Docs
predict_x=13
predict_dd_y=(m*predict_x)+b
plt.scatter(predict_x,predict_dd_y,color='yellow')
plt.show()


#regression line for source Code
m,b=best_fit_and_intercept(xs,ys_sc)
regression_line=[(m*x)+b for x in xs]

plt.scatter(xs,ys_sc,color='yellow')
plt.plot(xs,regression_line,color='black')

plt.title('SOURCE CODE VIOLATIONS')
plt.xlabel('MONTH', fontsize=18)
plt.ylabel('VIOLATIONS', fontsize=16)

#predict any month for source Code
predict_x=13
predict_sc_y=(m*predict_x)+b
plt.scatter(predict_x,predict_sc_y,color='red')
plt.show()

#regression line for Inventory Disclosure
m,b=best_fit_and_intercept(xs,ys_invd)
regression_line=[(m*x)+b for x in xs]

plt.scatter(xs,ys_invd,color='red')
plt.plot(xs,regression_line,color='black')
plt.title('INVENTORY DISCLOSURE VIOLATIONS')
plt.xlabel('MONTH', fontsize=18)
plt.ylabel('VIOLATIONS', fontsize=16)

#predict any month for Inventory Disclosure
predict_x=13
predict_invd_y=(m*predict_x)+b
plt.scatter(predict_x,predict_invd_y,color='blue')
plt.show()

#regression line for Program restricted Data
m,b=best_fit_and_intercept(xs,ys_prd)
regression_line=[(m*x)+b for x in xs]

plt.scatter(xs,ys_prd,color='orange')
plt.plot(xs,regression_line,color='black')

plt.title('PROGRAM RESTRICTED DATA VIOLATIONS')
plt.xlabel('MONTH', fontsize=18)
plt.ylabel('VIOLATIONS', fontsize=16)

#predict any month for Program restricted Data
predict_x=13
predict_prd_y=(m*predict_x)+b
plt.scatter(predict_x,predict_prd_y,color='blue')
plt.show()
#stores all the costs, it will be needed to find total cost

intermediate_costs=[]

#file path for summary sheet
pathout = '/home/jaxqueen/shruti/'
pathout+="estimated_cost_january.txt"
output=open(pathout,'a')
	
output.write("Cost Estimate Month of January")
output.write("\n")
output.write("\n")
output.write("Cost due to the following type of information security violation")
output.write("\n")
output.write("\n")
output.write("\n")

#fuction to write to file for each type
def filewrite(predict_type_y,costper_type):
	roundint=int(predict_type_y)
	output.write("%s" %predict_type_y)
	output.write("\n")
	output.write("Round off:"),
	output.write("%s" %roundint)
	output.write("\n")
	output.write("Cost per violation:"),
	output.write("$")
	output.write("%s" %costper_type),
	output.write("\n")
	output.write("Total:"),
	output.write("$")
	total=roundint*costper_type
	intermediate_costs.append(total)
	output.write("%s" %(total))
	output.write("\n")
	output.write("\n")
	output.write("\n")
	output.write("\n")
	


#cost estimates for month of january 

cost_types=np.array([predict_pii_y,predict_pci_y,predict_intd_y,predict_dd_y,predict_sc_y,predict_invd_y,predict_prd_y])

datakeycost = pd.read_csv('cost.csv', sep="\t", header = None,encoding='utf-8')
datakeycost.columns = ["email_cost","web_cost","usb_cost","print_cost","pii_cost","pci_cost","intd_cost","dd_cost","sc_cost","invd_cost","prd_cost"]
rowcountkey=len(datakeycost.index)


output.write("Predicted Number of violations due to PERSONAL IDENTITY INFORMATION VIOLATIONS:"),
filewrite(predict_pii_y,datakeycost.pii_cost[0])

output.write("Predicted Number of violations due to PERSONAL CREDIT INFORMATION VIOLATIONS:"),
filewrite(predict_pci_y,datakeycost.pci_cost[0])

output.write("Predicted Number of violations due to INTERNAL DOCUMENTS VIOLATIONS:"),
filewrite(predict_intd_y,datakeycost.intd_cost[0])

output.write("Predicted Number of violations due to DESIGN DOCUMENTS VIOLATIONS:"),
filewrite(predict_dd_y,datakeycost.dd_cost[0])

output.write("Predicted Number of violations due to SOURCE CODE VIOLATIONS:"),
filewrite(predict_sc_y,datakeycost.sc_cost[0])

output.write("Predicted Number of violations due to INVENTORY DISCLOSURE VIOLATIONS:"),
filewrite(predict_invd_y,datakeycost.invd_cost[0])

output.write("Predicted Number of violations due to PROGRAM RESTRICTED DATA VIOLATIONS:"),
filewrite(predict_prd_y,datakeycost.prd_cost[0])


#calculate total cost

total_cost=sum(intermediate_costs)
output.write("TOTAL COST DUE TO ABOVE TYPE OF VIOLATATIONS:"),
output.write("$")
output.write("%s" %total_cost)
output.write("\n")
output.write("\n")
output.write("\n")

pathout1 = '/home/jaxqueen/shruti/'
pathout1+="cost_inter.txt"
output1=open(pathout1,'a')
output1.write("%s" %total_cost)
output1.close()
		
